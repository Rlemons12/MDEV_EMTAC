# Kivey_UI/main_app.py

# region Todo: Clean up imports

# --- Standard Library Imports ---
import os
import sys
import json
import logging
from kivy.properties import ObjectProperty

# --- Kivy Core Imports ---
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import (
    StringProperty, ListProperty, BooleanProperty, NumericProperty
)
from kivy.metrics import dp
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.graphics import Color, Rectangle

# --- Kivy UI Widgets ---
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.modalview import ModalView
from kivy.uix.image import AsyncImage
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex

# --- KivyMD UI Widgets ---
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.list import (
    OneLineListItem, TwoLineListItem, ThreeLineListItem,
    OneLineIconListItem, IconLeftWidget
)
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase
from kivymd.uix.snackbar import MDSnackbar  # Consistent snackbar use

# --- Project Configuration ---
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger

# --- Database Models ---
from modules.emtacdb.emtacdb_fts import (
    Area, EquipmentGroup, Model, AssetNumber, Location,
    Position, Problem, Solution, Task, TaskSolutionAssociation,
    Part, Image, Drawing, CompleteDocument,KivyUser,
    DrawingPartAssociation, PartsPositionImageAssociation, ImagePositionAssociation,
    ProblemPositionAssociation, DrawingPositionAssociation, CompletedDocumentPositionAssociation
)

# --- UI Content Widgets ---
from modules.ui_emtac.content_widgets import (
    ProblemSolutionContent,
    PartsContent,
    DocumentsContent,
    ImagesContent,
    DrawingsContent
)

# --- Data Services ---
from data_service import DataService

from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.list import OneLineListItem
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager
from sqlalchemy.orm.exc import NoResultFound
from kivymd.uix.screen import MDScreen
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivy.metrics import dp
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDIconButton, MDRaisedButton
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.list import OneLineListItem, MDList
from kivymd.uix.dialog import MDDialog
from kivymd.uix.snackbar import MDSnackbar
from modules.emtacdb.emtacdb_fts import User
# endregion

# Add parent directory to path to find modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)
db_config = DatabaseConfig()
db_session = db_config.get_main_session()

# Set full window mode
Window.maximize()


class MainScreen(MDScreen):
    user = ObjectProperty(None)

    def __init__(self, user, **kwargs):
        super().__init__(**kwargs)
        self.name = "main_screen"
        self.user = user

        # Create main screen layout
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))

        # Create a welcome label with more detailed information
        welcome_text = self.get_welcome_message()
        welcome_label = MDLabel(
            text=welcome_text,
            halign='center',
            theme_text_color='Primary',
            font_style='H5'
        )
        layout.add_widget(welcome_label)



    def get_welcome_message(self):
        """Generate a personalized welcome message"""
        # Determine time of day for greeting
        import datetime
        current_hour = datetime.datetime.now().hour

        if current_hour < 12:
            time_greeting = "Good morning"
        elif 12 <= current_hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"

        # Create full name
        full_name = f"{self.user.first_name} {self.user.last_name}".strip()

        # Add user level if applicable
        user_level = getattr(self.user, 'user_level', None)
        level_text = f" ({user_level.value})" if user_level else ""

        return f"{time_greeting}, {full_name}{level_text}"

class PartDetailsPopup(ModalView):
    def __init__(self, part, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (0.9, 0.8)
        self.auto_dismiss = True
        self.background_color = (0, 0, 0, 0.5)

        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(15))

        # Header info
        layout.add_widget(MDLabel(text=f"[b]{part.part_number}[/b]", markup=True, font_style="H6"))
        layout.add_widget(MDLabel(text=f"Name: {part.name}"))
        layout.add_widget(MDLabel(text=f"Type: {part.type or 'N/A'}"))

        # Image
        if part.image_path and os.path.exists(part.image_path):
            layout.add_widget(AsyncImage(
                source=part.image_path,
                size_hint=(1, 0.6),
                allow_stretch=True
            ))
        else:
            layout.add_widget(MDLabel(text="No image available", halign="center"))

        # Close Button
        close_btn = MDRaisedButton(
            text="Close",
            pos_hint={"center_x": 0.5},
            on_release=self.dismiss
        )
        layout.add_widget(close_btn)

        self.add_widget(layout)

class MainLayout(MDFloatLayout):
    """Main layout container for the application"""
    pass

class NavigationItem(OneLineIconListItem):
    """Navigation menu item with icon"""
    icon = StringProperty("")

class ResizableDraggableCard(MDCard):
    is_expanded = BooleanProperty(False)
    original_pos = ListProperty([0, 0])
    original_size = ListProperty([0, 0])
    title = StringProperty("Panel")

    min_width = NumericProperty(dp(300))
    min_height = NumericProperty(dp(250))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self._store_initial_values, 0.1)
        self.bind(pos=self._update_resize_handle, size=self._update_resize_handle)
        self.bind(size=self._update_content_layout)

        # Define header height for dragging
        self.header_height = dp(40)

    def _store_initial_values(self, *_):
        self.original_pos = self.pos
        self.original_size = self.size

    def _update_resize_handle(self, *_):
        try:
            if hasattr(self, 'ids') and 'resize_handle' in self.ids:
                self.ids.resize_handle.pos = (self.right - dp(20), self.y)
            elif hasattr(self, 'resize_handle'):
                self.resize_handle.pos = (self.right - dp(20), self.y)
        except (ReferenceError, AttributeError):
            pass

    def _update_content_layout(self, *_):
        Clock.schedule_once(self._force_content_update, 0)

    def _force_content_update(self, dt):
        try:
            for child in self.children:
                self._process_widget_layout(child)
        except ReferenceError:
            pass

    def _process_widget_layout(self, widget):
        if not widget:
            return
        try:
            if hasattr(widget, 'do_layout'):
                widget.do_layout()
            if isinstance(widget, Label):
                widget.text_size = (widget.width, None)
                widget.texture_update()
                if widget.texture:
                    widget.height = max(dp(30), widget.texture.height)
            if hasattr(widget, 'children'):
                for child in widget.children:
                    self._process_widget_layout(child)
        except ReferenceError:
            pass

    def collides_with_other(self, new_x, new_y, new_width=None, new_height=None):
        """
        Checks if a candidate rectangle (position and size) overlaps with any other panel.
        """
        if new_width is None:
            new_width = self.width
        if new_height is None:
            new_height = self.height

        # Candidate rectangle: (left, bottom, right, top)
        candidate = (new_x, new_y, new_x + new_width, new_y + new_height)
        app = MDApp.get_running_app()
        for panel in app.get_all_panels():
            if panel is self:
                continue
            panel_rect = (panel.x, panel.y, panel.x + panel.width, panel.y + panel.height)
            if (candidate[2] > panel_rect[0] and candidate[0] < panel_rect[2] and
                    candidate[3] > panel_rect[1] and candidate[1] < panel_rect[3]):
                return True
        return False

    def on_touch_down(self, touch):
        # Check if we're in the header area (for dragging)
        is_in_header = False
        if self.collide_point(*touch.pos):
            # Check if touch is in the title bar area
            header_area = self.y + self.height - self.header_height
            if touch.y >= header_area:
                is_in_header = True
                touch.grab(self)
                touch.ud['drag'] = True
                return True

            # Check if we're in resize area
            if self._in_resize_area(touch.pos):
                touch.grab(self)
                touch.ud['resize'] = True
                return True

        # If not in header or resize area, let children handle the touch
        if self.collide_point(*touch.pos) and not is_in_header:
            # First try to dispatch to children
            for child in self.children:
                if child.dispatch('on_touch_down', touch):
                    return True

        # Fallback to super's behavior
        return super().on_touch_down(touch)

    def _in_resize_area(self, pos):
        resize_zone_size = dp(30)
        x_in_zone = self.right - resize_zone_size <= pos[0] <= self.right
        y_in_zone = self.y <= pos[1] <= self.y + resize_zone_size
        return x_in_zone and y_in_zone

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            if touch.ud.get('resize', False):
                self._perform_resize(touch)
                return True
            elif touch.ud.get('drag', False):
                self._perform_drag(touch)
                return True

        # Let children handle the touch if not being dragged or resized
        for child in self.children:
            if child.dispatch('on_touch_move', touch):
                return True

        return super().on_touch_move(touch)

    def _perform_drag(self, touch):
        window_width, window_height = Window.size
        top_banner = dp(60)
        candidate_x = min(max(0, self.x + touch.dx), window_width - self.width)
        candidate_y = min(max(0, self.y + touch.dy), window_height - top_banner - self.height)
        if not self.collides_with_other(candidate_x, self.y):
            self.x = candidate_x
        if not self.collides_with_other(self.x, candidate_y):
            self.y = candidate_y

    def _perform_resize(self, touch):
        min_width = self.min_width
        min_height = self.min_height
        window_width, window_height = Window.size
        top_banner = dp(60)
        candidate_width = min(max(min_width, self.width + touch.dx), window_width - self.x)
        candidate_height = min(max(min_height, self.height - touch.dy), window_height - top_banner - self.y)
        candidate_y = self.y + touch.dy if candidate_height > min_height else self.y
        if candidate_y < 0:
            candidate_y = 0
        if not self.collides_with_other(self.x, candidate_y, candidate_width, candidate_height):
            self.size = (candidate_width, candidate_height)
            self.y = candidate_y

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            self._clamp_to_window()
            return True

        # Let children handle the touch
        for child in self.children:
            if child.dispatch('on_touch_up', touch):
                return True

        return super().on_touch_up(touch)

    def _clamp_to_window(self):
        window_width, window_height = Window.size
        top_banner = dp(60)
        self.x = min(max(0, self.x), window_width - self.width)
        self.y = min(max(0, self.y), window_height - top_banner - self.height)

    def toggle_expand(self):
        app = MDApp.get_running_app()
        if not self.is_expanded:
            self.original_pos = self.pos
            self.original_size = self.size
            self.pos = (dp(10), dp(10))
            self.size = (Window.width * 0.8, Window.height * 0.85)
        else:
            self.pos = self.original_pos
            self.size = self.original_size
        self.is_expanded = not self.is_expanded
        Clock.schedule_once(self._force_content_update, 0.1)

class NavigationSpinner(Spinner):
    """Spinner for equipment hierarchy navigation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(40)
        self.sync_height = True

class Tab(MDFloatLayout, MDTabsBase):
    """Class implementing content for a tab."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tab_label = kwargs.get('text', '')

class LoginScreen(MDScreen):
    def __init__(self, login_callback, **kwargs):
        super().__init__(**kwargs)
        self.login_callback = login_callback
        self.name = "login_screen"

        # Dark grey background
        layout = MDBoxLayout(
            orientation='vertical',
            padding=dp(40),
            spacing=dp(25),
            size_hint=(None, None),
            size=(dp(450), dp(400)),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            md_bg_color=get_color_from_hex("#424242")
        )

        # Title
        title = MDLabel(
            text="Maintenance Troubleshooting",
            font_style="H5",
            halign="center",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#FFFFFF"),
            size_hint_y=None,
            height=dp(40)
        )
        layout.add_widget(title)

        # Subtitle
        subtitle = MDLabel(
            text="Login",
            font_style="H6",
            halign="center",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#E0E0E0"),
            size_hint_y=None,
            height=dp(30)
        )
        layout.add_widget(subtitle)

        # Employee ID field
        id_layout = MDBoxLayout(orientation='vertical', size_hint_y=None, height=dp(70))
        id_label = MDLabel(
            text="Employee ID",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#CCCCCC"),
            size_hint_y=None,
            height=dp(20)
        )
        self.employee_id = MDTextField(
            hint_text="Enter your employee ID",
            mode="rectangle",
            size_hint_y=None,
            height=dp(50),
            line_color_normal=get_color_from_hex("#64B5F6"),
        )
        # now apply the colors that weren't accepted as kwargs
        self.employee_id.foreground_color = get_color_from_hex("#FFFFFF")
        self.employee_id.hint_text_color   = get_color_from_hex("#BBBBBB")

        id_layout.add_widget(id_label)
        id_layout.add_widget(self.employee_id)
        layout.add_widget(id_layout)

        # Password field
        pass_layout = MDBoxLayout(orientation='vertical', size_hint_y=None, height=dp(70))
        pass_label = MDLabel(
            text="Password",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#CCCCCC"),
            size_hint_y=None,
            height=dp(20)
        )
        self.password = MDTextField(
            hint_text="Enter your password",
            password=True,
            mode="rectangle",
            size_hint_y=None,
            height=dp(50),
            line_color_normal=get_color_from_hex("#64B5F6"),
        )
        self.password.foreground_color = get_color_from_hex("#FFFFFF")
        self.password.hint_text_color   = get_color_from_hex("#BBBBBB")

        pass_layout.add_widget(pass_label)
        pass_layout.add_widget(self.password)
        layout.add_widget(pass_layout)

        # Spacer
        layout.add_widget(Widget(size_hint_y=None, height=dp(10)))

        # Login button
        login_button = MDRaisedButton(
            text="Login",
            font_style="Button",
            size_hint=(1, None),
            height=dp(50),
            md_bg_color=get_color_from_hex("#1976D2"),
            text_color=get_color_from_hex("#FFFFFF"),
            on_release=self.attempt_login
        )
        layout.add_widget(login_button)

        # Error label
        self.error_label = MDLabel(
            text="",
            theme_text_color="Custom",
            text_color=get_color_from_hex("#EF5350"),
            halign="center",
            size_hint_y=None,
            height=dp(30),
            font_style="Body1",
            bold=True
        )
        layout.add_widget(self.error_label)

        self.add_widget(layout)

    def attempt_login(self, instance):
        employee_id = self.employee_id.text.strip()
        password = self.password.text

        if not employee_id or not password:
            self.error_label.text = "Please enter both Employee ID and Password"
            return

        # Authenticate against the database
        try:
            # Get the database session
            session = MDApp.get_running_app().data_service.session

            # Query the user by employee_id
            from modules.emtacdb.emtacdb_fts import User

            # Use first() instead of one() to handle "not found" more gracefully
            user = session.query(User).filter(User.employee_id == employee_id).first()

            if not user:
                self.error_label.text = "Employee ID not found"
                return

            # Check the password using the method from the User model
            if user.check_password_hash(password):
                # Success - call the login callback with the user object
                self.login_callback(user)
            else:
                self.error_label.text = "Invalid password"

        except Exception as e:
            print(f"Login error: {e}")
            import traceback
            traceback.print_exc()  # More detailed error logging
            self.error_label.text = "An error occurred during login"

class MaintenanceTroubleshootingApp(MDApp):
    current_position_id = NumericProperty(-1)  # Using -1 as sentinel value
    current_user = None  # Store the logged-in user

    def build(self):
        # 1) Load your KV before instantiating any screens
        Builder.load_file('maintenance_ui.kv')

        # 2) Set up your theme
        self.theme_cls.primary_palette = "Cyan"
        self.theme_cls.accent_palette = "Amber"
        self.theme_cls.theme_style   = "Dark"

        # 3) Initialize your data service
        self.data_service = DataService(db_session)

        # 4) Create and configure the ScreenManager
        self.screen_manager = ScreenManager()

        # 5) Add only the login screen for now
        self.login_screen = LoginScreen(login_callback=self.on_login_success)
        self.screen_manager.add_widget(self.login_screen)

        # 6) Don’t add MainScreen here—do that in on_login_success
        return self.screen_manager

    def on_login_success(self, user):
        """Called when login is successful"""
        from sqlalchemy import text
        session = self.data_service.session
        kivy_user = session.query(KivyUser).filter(KivyUser.id == user.id).first()

        # Your existing code for creating/fetching KivyUser...

        # Store the current user (either KivyUser or regular User)
        self.current_user = kivy_user
        print(f"User logged in: {kivy_user.first_name} {kivy_user.last_name} (ID: {kivy_user.employee_id})")
        print(f"User type: {type(kivy_user).__name__}")

        # 2) Instantiate MainScreen (will pick up your <MainScreen> KV rule)
        self.main_screen = MainScreen(user=kivy_user)  # Pass KivyUser instead of base User

        # 3) Add MainScreen if not already in the manager
        if not any(screen.name == "main_screen" for screen in self.screen_manager.screens):
            self.screen_manager.add_widget(self.main_screen)

        # 4) Switch to MainScreen
        self.screen_manager.current = "main_screen"
        print("Screens:", [s.name for s in self.screen_manager.screens])
        print("Current screen:", self.screen_manager.current)

        # 5) Schedule panel content setup
        Clock.schedule_once(self.initialize_panel_content, 0.5)

        # 6) Schedule populating the navigation drawer
        Clock.schedule_once(self.add_equipment_navigation, 0.5)

        # 7) Initialize default layouts
        Clock.schedule_once(lambda dt: self.initialize_default_layouts(), 1.0)

        # 8) Apply default layout immediately after initialization
        Clock.schedule_once(lambda dt: self.apply_default_layout(), 1.2)

    def on_start(self):
        # This method will no longer do anything since initialization happens in on_login_success
        pass

    def on_stop(self):
        try:
            db_session.close()
        except:
            pass

    def initialize_panel_content(self, dt):
        try:
            panels = self.get_all_panels()

            # Create shared widget instances
            drawing_widget = DrawingsContent()
            parts_widget = PartsContent()
            image_widget = ImagesContent()  # create instance of ImageContent

            for panel in panels:
                if panel.title == "Problem / Solution":
                    panel.clear_widgets()
                    panel.add_widget(
                        ProblemSolutionContent(
                            drawing_content=drawing_widget,
                            parts_content=parts_widget
                        )
                    )
                elif panel.title == "Documents":
                    panel.clear_widgets()
                    self.documents_content = DocumentsContent()
                    panel.add_widget(self.documents_content)

                elif panel.title == "Parts":
                    panel.clear_widgets()
                    panel.add_widget(parts_widget)

                elif panel.title == "Images":
                    panel.clear_widgets()
                    panel.add_widget(image_widget)  # Use ImageContent instance

                elif panel.title == "Drawings":
                    panel.clear_widgets()
                    panel.add_widget(drawing_widget)

            # Optionally store for access
            self.image_content_widget = image_widget

        except Exception as e:
            logger.exception(f"Error initializing panel content: {e}")

    def add_equipment_navigation(self, dt):
        try:
            print("Adding equipment navigation...")  # Debugging

            # Try both methods to get the navigation drawer
            nav_drawer = self.root.ids.get('nav_drawer', None)
            if not nav_drawer:
                # Try alternate method
                main = self.screen_manager.get_screen("main_screen")
                nav_drawer = main.ids.get('nav_drawer')
                if not nav_drawer:
                    print("Navigation drawer not found")
                    return

            # Create a main layout for the drawer with same spacing as original
            equipment_nav = MDBoxLayout(orientation='vertical', spacing=dp(5), padding=dp(10))

            # Add a simple search box with a Search button and a Clear button
            search_layout = MDBoxLayout(
                orientation='horizontal',
                spacing=dp(5),
                size_hint_y=None,
                height=dp(48)
            )

            # Create a text field for asset number entry
            self.asset_search_field = MDTextField(
                hint_text="Search Asset Number",
                size_hint_x=0.6
            )
            self.asset_search_field.bind(text=self.on_asset_search_text)

            # Create a Search button
            search_button = MDRaisedButton(
                text="Search",
                size_hint_x=0.2,
                on_release=self.on_search_button_click
            )

            # Create a Clear button for the search field
            clear_button = MDRaisedButton(
                text="Clear",
                size_hint_x=0.2,
                on_release=self.on_clear_button_click
            )

            # Add widgets to the search layout
            search_layout.add_widget(self.asset_search_field)
            search_layout.add_widget(search_button)
            search_layout.add_widget(clear_button)

            # Add spacing above and below the search layout
            equipment_nav.add_widget(MDBoxLayout(size_hint_y=None, height=dp(10)))
            equipment_nav.add_widget(search_layout)
            equipment_nav.add_widget(MDBoxLayout(size_hint_y=None, height=dp(10)))

            # Add remaining navigation components (areas, groups, etc.)
            # Use the appropriate Spinner class - check if NavigationSpinner is defined
            try:
                SpinnerClass = NavigationSpinner
            except NameError:
                from kivymd.uix.spinner import MDSpinner as SpinnerClass

            areas_label = Label(text="Areas:", size_hint_y=None, height=dp(30))
            self.areas_spinner = SpinnerClass(text="Select an area")
            groups_label = Label(text="Equipment Groups:", size_hint_y=None, height=dp(30))
            self.groups_spinner = SpinnerClass(text="Select a group")
            models_label = Label(text="Models:", size_hint_y=None, height=dp(30))
            self.models_spinner = SpinnerClass(text="Select a model")
            locations_label = Label(text="Locations:", size_hint_y=None, height=dp(30))
            self.locations_spinner = SpinnerClass(text="Select a location")

            # Position control buttons
            position_controls = MDBoxLayout(
                orientation='horizontal',
                spacing=dp(10),
                size_hint_y=None,
                height=dp(40)
            )

            set_position_btn = MDRaisedButton(
                text="Set Position",
                size_hint=(None, None),
                size=(dp(150), dp(40)),
                pos_hint={'center_x': 0.5}
            )
            set_position_btn.bind(on_release=self.set_current_position)

            clear_position_btn = MDRaisedButton(
                text="Clear Position",
                size_hint=(None, None),
                size=(dp(150), dp(40)),
                pos_hint={'center_x': 0.5}
            )
            clear_position_btn.bind(on_release=self.on_clear_position_click)

            position_controls.add_widget(set_position_btn)
            position_controls.add_widget(clear_position_btn)

            # Layout control buttons
            layout_controls = MDBoxLayout(orientation='horizontal', spacing=dp(10), size_hint_y=None, height=dp(40))
            save_layout_btn = MDRaisedButton(
                text="Save Layout",
                size_hint=(None, None),
                size=(dp(120), dp(40))
            )
            load_layout_btn = MDRaisedButton(
                text="Load Layout",
                size_hint=(None, None),
                size=(dp(120), dp(40))
            )
            delete_layout_btn = MDRaisedButton(
                text="Delete Layout",
                size_hint=(None, None),
                size=(dp(120), dp(40))
            )
            default_layout_btn = MDRaisedButton(
                text="Default Layout",
                size_hint=(None, None),
                size=(dp(120), dp(40))
            )

            save_layout_btn.bind(on_release=lambda x: self.prompt_save_layout())
            load_layout_btn.bind(on_release=lambda x: self.load_layout())
            delete_layout_btn.bind(on_release=lambda x: self.delete_layout())
            default_layout_btn.bind(on_release=lambda x: self.load_layout("Default"))

            layout_controls.add_widget(save_layout_btn)
            layout_controls.add_widget(load_layout_btn)
            layout_controls.add_widget(delete_layout_btn)
            layout_controls.add_widget(default_layout_btn)

            # Spinner for saved layouts
            self.layout_spinner = SpinnerClass(text="Select Layout")

            # Add all widgets to the navigation drawer
            equipment_nav.add_widget(areas_label)
            equipment_nav.add_widget(self.areas_spinner)
            equipment_nav.add_widget(groups_label)
            equipment_nav.add_widget(self.groups_spinner)
            equipment_nav.add_widget(models_label)
            equipment_nav.add_widget(self.models_spinner)
            equipment_nav.add_widget(locations_label)
            equipment_nav.add_widget(self.locations_spinner)
            equipment_nav.add_widget(position_controls)
            equipment_nav.add_widget(layout_controls)
            equipment_nav.add_widget(self.layout_spinner)

            # Bindings for the spinners
            self.areas_spinner.bind(text=self.on_area_selected)
            self.groups_spinner.bind(text=self.on_group_selected)
            self.models_spinner.bind(text=self.on_model_selected)

            # Clear any existing widgets and add our new layout
            nav_drawer.clear_widgets()
            nav_drawer.add_widget(equipment_nav)

            # Load the dropdown values
            self.load_areas()
            self.update_layout_spinner()

            print("Equipment navigation setup complete")  # Debugging

        except Exception as e:
            print(f"Error adding equipment navigation: {e}")
            import traceback
            traceback.print_exc()

    def on_search_button_click(self, instance):
        """Handle asset number search button clicks"""
        search_term = self.asset_search_field.text
        print(f"Search button clicked with term: '{search_term}'")

        if not search_term:
            print("Empty search term")
            snackbar = MDSnackbar()
            snackbar.text = "Please enter an asset number to search"
            snackbar.open()
            return

        try:
            # Use the get_ids_by_number method to find asset numbers
            print(f"Searching for asset number: {search_term}")
            asset_ids = AssetNumber.get_ids_by_number(self.data_service.session, search_term)

            print(f"Found {len(asset_ids)} matching assets")

            if not asset_ids:
                snackbar = MDSnackbar()
                snackbar.text = f"No asset numbers found matching '{search_term}'"
                snackbar.open()
                return

            # Get the full AssetNumber objects for display
            assets = self.data_service.session.query(AssetNumber).filter(AssetNumber.id.in_(asset_ids)).all()
            print(f"Retrieved {len(assets)} asset objects from database")

            # Create a simple content layout
            from kivymd.uix.boxlayout import MDBoxLayout
            from kivymd.uix.button import MDRaisedButton
            from kivy.metrics import dp

            content_layout = MDBoxLayout(
                orientation='vertical',
                spacing=dp(10),
                padding=dp(10),
                size_hint_y=None,
                height=dp(60 * len(assets) + 20)  # Height based on number of assets
            )

            # Store reference to the class instance
            app_instance = self

            # Add each asset as a button
            for asset in assets:
                model = self.data_service.session.query(Model).filter(Model.id == asset.model_id).first()
                model_name = model.name if model else "Unknown model"

                print(f"Creating button for asset #{asset.number} - {model_name}")

                asset_btn = MDRaisedButton(
                    text=f"Asset #{asset.number} - {model_name}",
                    size_hint_x=1,
                    height=dp(50)
                )

                # Store the asset with the button
                asset_btn.asset_data = asset


                asset_btn.bind(on_release=self.on_asset_button_click)

                content_layout.add_widget(asset_btn)

            # Create and show a simple dialog
            print("Creating dialog with asset buttons")
            self.search_results_dialog = MDDialog(
                title=f"Search results for '{search_term}'",
                type="custom",
                content_cls=content_layout,
                size_hint=(0.8, None),
                height=dp(100 + 60 * len(assets)),
                buttons=[
                    MDRaisedButton(
                        text="Close",
                        on_release=lambda x: self.search_results_dialog.dismiss()
                    )
                ]
            )
            print("Opening dialog")
            self.search_results_dialog.open()
            print("Dialog opened")

        except Exception as e:
            print(f"Error searching for asset number: {e}")
            import traceback
            traceback.print_exc()
            snackbar = MDSnackbar()
            snackbar.text = f"Error during search: {e}"
            snackbar.open()

    def load_areas(self):
        """Load all areas into the areas spinner"""
        try:
            print("Loading areas...")
            areas = self.data_service.get_all_areas()
            if areas:
                self.areas_spinner.values = [area.name for area in areas]
                print(f"Loaded {len(areas)} areas")
            else:
                print("No areas found")
                self.areas_spinner.values = []
        except Exception as e:
            print(f"Error loading areas: {e}")
            import traceback
            traceback.print_exc()

    def on_area_selected(self, spinner, text):
        if text == "Select an area":
            return
        try:
            areas = self.data_service.get_all_areas()
            area = next((a for a in areas if a.name == text), None)
            if area:
                groups = self.data_service.get_equipment_groups_by_area(area.id)
                if groups:
                    self.groups_spinner.values = [group.name for group in groups]
                    self.groups_spinner.text = "Select a group"
                else:
                    self.groups_spinner.values = []
                    self.groups_spinner.text = "No groups found"
                self.models_spinner.values = []
                self.models_spinner.text = "Select a model"
                self.locations_spinner.values = []
                self.locations_spinner.text = "Select a location"
        except Exception as e:
            print(f"Error on area selection: {e}")

    def on_group_selected(self, spinner, text):
        if text == "Select a group" or text == "No groups found":
            return
        try:
            area = next((a for a in self.data_service.get_all_areas()
                         if a.name == self.areas_spinner.text), None)
            if not area:
                return
            groups = self.data_service.get_equipment_groups_by_area(area.id)
            group = next((g for g in groups if g.name == text), None)
            if group:
                models = self.data_service.get_models_by_equipment_group(group.id)
                if models:
                    self.models_spinner.values = [model.name for model in models]
                    self.models_spinner.text = "Select a model"
                else:
                    self.models_spinner.values = []
                    self.models_spinner.text = "No models found"
                self.locations_spinner.values = []
                self.locations_spinner.text = "Select a location"
        except Exception as e:
            print(f"Error on group selection: {e}")

    def on_model_selected(self, spinner, text):
        if text == "Select a model" or text == "No models found":
            return
        try:
            area = next((a for a in self.data_service.get_all_areas()
                         if a.name == self.areas_spinner.text), None)
            if not area:
                return
            group = next((g for g in self.data_service.get_equipment_groups_by_area(area.id)
                          if g.name == self.groups_spinner.text), None)
            if not group:
                return
            models = self.data_service.get_models_by_equipment_group(group.id)
            model = next((m for m in models if m.name == text), None)
            if model:
                locations = self.data_service.get_locations_by_model(model.id)
                if locations:
                    self.locations_spinner.values = [location.name for location in locations]
                    self.locations_spinner.text = "Select a location"
                else:
                    self.locations_spinner.values = []
                    self.locations_spinner.text = "No locations found"
        except Exception as e:
            print(f"Error on model selection: {e}")

    def set_current_position(self, button):
        try:
            area = next((a for a in self.data_service.get_all_areas()
                         if a.name == self.areas_spinner.text), None)
            area_id = None
            if area and self.areas_spinner.text != "Select an area":
                area_id = area.id

            group_id = None
            if area_id and self.groups_spinner.text != "Select a group" and self.groups_spinner.text != "No groups found":
                group = next((g for g in self.data_service.get_equipment_groups_by_area(area_id)
                              if g.name == self.groups_spinner.text), None)
                if group:
                    group_id = group.id

            model_id = None
            if group_id and self.models_spinner.text != "Select a model" and self.models_spinner.text != "No models found":
                model = next((m for m in self.data_service.get_models_by_equipment_group(group_id)
                              if m.name == self.models_spinner.text), None)
                if model:
                    model_id = model.id

            location_id = None
            if model_id and self.locations_spinner.text != "Select a location" and self.locations_spinner.text != "No locations found":
                location = next((l for l in self.data_service.get_locations_by_model(model_id)
                                 if l.name == self.locations_spinner.text), None)
                if location:
                    location_id = location.id

            # Store the hierarchy IDs for use in refreshing
            self.current_area_id = area_id
            self.current_group_id = group_id
            self.current_model_id = model_id
            self.current_location_id = location_id

            # If we have at least an area selected, set the filters and refresh
            if area_id:
                # Add diagnostic section before refreshing panels
                print("\n----- DIAGNOSTIC OUTPUT START -----")
                print(f"Selected filters: Area={area_id}, Group={group_id}, Model={model_id}, Location={location_id}")

                # Find positions matching these filters
                positions = self.data_service.session.query(Position.id).distinct()
                if area_id:
                    positions = positions.filter(Position.area_id == area_id)
                if group_id:
                    positions = positions.filter(Position.equipment_group_id == group_id)
                if model_id:
                    positions = positions.filter(Position.model_id == model_id)
                if location_id:
                    positions = positions.filter(Position.location_id == location_id)

                position_ids = [pos_id for pos_id, in positions.all()]
                print(f"Found {len(position_ids)} positions: {position_ids}")

                # Check for problems for these positions
                if position_ids:
                    for pos_id in position_ids:
                        print(f"\nChecking position ID: {pos_id}")
                        problems = self.data_service.get_problems_by_position(pos_id)
                        print(f"Found {len(problems)} problems for this position")

                        for problem in problems:
                            print(f"\nProblem ID: {problem.id}, Name: {problem.name}")
                            # Check for solutions
                            solutions = self.data_service.session.query(Solution).filter(
                                Solution.problem_id == problem.id).all()
                            print(f"Found {len(solutions)} solutions for Problem ID {problem.id}")

                            for solution in solutions:
                                print(f"  Solution ID: {solution.id}, Name: {solution.name}")
                                # Check for tasks
                                task_assocs = self.data_service.session.query(TaskSolutionAssociation).filter(
                                    TaskSolutionAssociation.solution_id == solution.id).all()
                                print(f"  Found {len(task_assocs)} task associations")

                                for assoc in task_assocs:
                                    task = self.data_service.session.query(Task).filter(
                                        Task.id == assoc.task_id).first()
                                    if task:
                                        print(f"    Task ID: {task.id}, Name: {task.name}")

                print("----- DIAGNOSTIC OUTPUT END -----\n")

                # Continue with normal operation
                self.refresh_panel_content_with_filters()
                self.root.ids.nav_drawer.set_state("close")

                # Create description of what we're viewing
                description_parts = []
                if area_id:
                    description_parts.append(f"Area: {self.areas_spinner.text}")
                if group_id:
                    description_parts.append(f"Group: {self.groups_spinner.text}")
                if model_id:
                    description_parts.append(f"Model: {self.models_spinner.text}")
                if location_id:
                    description_parts.append(f"Location: {self.locations_spinner.text}")

                hierarchy_desc = ", ".join(description_parts)

                snackbar = MDSnackbar()
                snackbar.text = f"Viewing data for {hierarchy_desc}"
                snackbar.open()
            else:
                snackbar = MDSnackbar()
                snackbar.text = "Please select at least an area"
                snackbar.open()

        except Exception as e:
            print(f"Error setting hierarchy: {e}")
            import traceback
            traceback.print_exc()  # This will print the full stack trace
            snackbar = MDSnackbar()
            snackbar.text = f"Error setting hierarchy: {e}"
            snackbar.open()

    def refresh_panel_content(self):
        if self.current_position_id == -1:  # Changed from "if not self.current_position_id:"
            return
        try:
            panels = self.get_all_panels()
            for panel in panels:
                if not panel.children:
                    continue
                content_widget = panel.children[0]
                if isinstance(content_widget, ProblemSolutionContent):
                    # This widget uses problem_container internally.
                    content_widget.update_for_position(self.current_position_id)
                elif isinstance(content_widget, PartsContent):
                    # This widget uses parts_container internally.
                    content_widget.update_for_position(self.current_position_id)
                elif isinstance(content_widget, DocumentsContent):
                    content_widget.update_for_position(self.current_position_id)
                elif isinstance(content_widget, ImagesContent):
                    content_widget.update_for_position(self.current_position_id)
                elif isinstance(content_widget, DrawingsContent):
                    content_widget.update_for_position(self.current_position_id)
                else:
                    # Fallback in case another widget type implements update_for_position.
                    if hasattr(content_widget, 'update_for_position'):
                        content_widget.update_for_position(self.current_position_id)
            # End for
        except Exception as e:
            print(f"Error refreshing panel content: {e}")

    def toggle_nav_drawer(self):
        main = self.screen_manager.get_screen("main_screen")
        nav = main.ids.nav_drawer
        nav.set_state("open" if nav.state == "close" else "close")

    def apply_default_layout(self):
        """Reset the layout to default positions"""
        try:
            panels = self.get_all_panels()

            # Use exact dimensions from your JSON
            layout_config = {
                "Problem / Solution": {"x": 10.0, "y": 10.0, "width": 940.0, "height": 929.0, "is_expanded": False},
                "Documents": {"x": 960.0, "y": 489.5, "width": 470.0, "height": 459.5, "is_expanded": False},
                "Parts": {"x": 1440.0, "y": 489.5, "width": 470.0, "height": 459.5, "is_expanded": False},
                "Images": {"x": 960.0, "y": 10.0, "width": 470.0, "height": 459.5, "is_expanded": False},
                "Drawings": {"x": 1440.0, "y": 10.0, "width": 470.0, "height": 459.5, "is_expanded": False}
            }

            for panel in panels:
                if panel.title in layout_config:
                    config = layout_config[panel.title]
                    panel.size = (config["width"], config["height"])
                    panel.pos = (config["x"], config["y"])

                    # Set expansion state if needed
                    if config["is_expanded"] != panel.is_expanded:
                        panel.toggle_expand()

            Clock.schedule_once(self.update_canvas, 0.1)

        except Exception as e:
            import traceback
            print(f"Error applying default layout: {e}")
            traceback.print_exc()

    def update_canvas(self, dt=None):
        if hasattr(self.root, 'canvas'):
            self.root.canvas.ask_update()

    def on_window_resize(self, instance, width, height):
        Clock.unschedule(self._adjust_panels_after_resize)
        Clock.schedule_once(self._adjust_panels_after_resize, 0.1)

    def _adjust_panels_after_resize(self, dt):
        panels = self.get_all_panels()
        self._ensure_panels_within_bounds(panels)
        self._resolve_panel_overlaps(panels)

    def get_all_panels(self):
        """Retrieve all panel widgets"""
        panels = []
        try:
            for child in self.root.walk():
                if hasattr(child, 'title') and isinstance(child, ResizableDraggableCard) and not child.is_expanded:
                    panels.append(child)
        except Exception as e:
            print(f"Error finding panels: {e}")
        return panels

    def _ensure_panels_within_bounds(self, panels):
        window_width = self.root.width
        window_height = self.root.height
        margin = dp(5)
        banner_height = dp(60)
        for panel in panels:
            if panel.x + panel.width > window_width - margin:
                panel.x = max(margin, window_width - panel.width - margin)
            if panel.y < margin:
                panel.y = margin
            if panel.x < margin:
                panel.x = margin
            if panel.y + panel.height > window_height - banner_height - margin:
                panel.y = window_height - banner_height - panel.height - margin

    def _resolve_panel_overlaps(self, panels):
        sorted_panels = sorted(panels, key=lambda p: p.width * p.height, reverse=True)
        for i, panel1 in enumerate(sorted_panels):
            for panel2 in sorted_panels[i + 1:]:
                if self._panels_overlap(panel1, panel2):
                    self._separate_panels(panel1, panel2)

    def _panels_overlap(self, panel1, panel2):
        left1, right1 = panel1.x, panel1.x + panel1.width
        top1, bottom1 = panel1.y + panel1.height, panel1.y
        left2, right2 = panel2.x, panel2.x + panel2.width
        top2, bottom2 = panel2.y + panel2.height, panel2.y
        return (right1 > left2 and left1 < right2 and
                top1 > bottom2 and bottom1 < top2)

    def _separate_panels(self, panel1, panel2):
        left_overlap = (panel1.x + panel1.width) - panel2.x
        right_overlap = (panel2.x + panel2.width) - panel1.x
        top_overlap = (panel1.y + panel1.height) - panel2.y
        bottom_overlap = (panel2.y + panel2.height) - panel1.y
        min_overlap = min(left_overlap, right_overlap, top_overlap, bottom_overlap)
        window_width = self.root.width
        window_height = self.root.height
        margin = dp(5)
        banner_height = dp(60)
        if min_overlap == left_overlap:
            new_x = panel1.x + panel1.width + margin
            if new_x + panel2.width <= window_width - margin:
                panel2.x = new_x
            else:
                new_x = panel1.x - panel2.width - margin
                if new_x >= margin:
                    panel2.x = new_x
        elif min_overlap == right_overlap:
            new_x = panel1.x - panel2.width - margin
            if new_x >= margin:
                panel2.x = new_x
            else:
                new_x = panel1.x + panel1.width + margin
                if new_x + panel2.width <= window_width - margin:
                    panel2.x = new_x
        elif min_overlap == top_overlap:
            new_y = panel1.y - panel2.height - margin
            if new_y >= margin:
                panel2.y = new_y
            else:
                new_y = panel1.y + panel1.height + margin
                if new_y + panel2.height <= window_height - banner_height - margin:
                    panel2.y = new_y
        elif min_overlap == bottom_overlap:
            new_y = panel1.y + panel1.height + margin
            if new_y + panel2.height <= window_height - banner_height - margin:
                panel2.y = new_y
            else:
                new_y = panel1.y - panel2.height - margin
                if new_y >= margin:
                    panel2.y = new_y

    def prompt_save_layout(self):
        """Prompt user to save current layout configuration"""
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.textfield import MDTextField

        content = MDBoxLayout(orientation="vertical", spacing=dp(20), padding=dp(20))
        self.layout_name_field = MDTextField(hint_text="Enter layout name")
        content.add_widget(self.layout_name_field)

        self.save_dialog = MDDialog(
            title="Save Layout",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: self.save_dialog.dismiss()),
                MDRaisedButton(text="Save", on_release=self.on_save_layout_confirm)
            ]
        )
        self.save_dialog.open()

    def on_save_layout_confirm(self, instance):
        """Confirm and save the current layout"""
        layout_name = self.layout_name_field.text.strip()
        if layout_name:
            self._save_layout(layout_name)
            self.save_dialog.dismiss()
        else:
            snackbar = MDSnackbar()
            snackbar.text = "Please enter a valid layout name"
            snackbar.open()

    def _save_layout(self, layout_name):
        """Save the current layout configuration to the database using KivyUser"""
        try:
            # Check if current user is a KivyUser
            if not isinstance(self.current_user, KivyUser):
                # If not, try to convert the user to a KivyUser on the fly
                session = self.data_service.session

                try:
                    from sqlalchemy import text
                    # First check if there's already a kivy_user entry for this user
                    kivy_check = session.execute(
                        text("SELECT id FROM kivy_users WHERE id = :user_id"),
                        {"user_id": self.current_user.id}
                    ).fetchone()

                    if not kivy_check:
                        # Try to add a kivy_user entry
                        session.execute(
                            text("INSERT INTO kivy_users (id) VALUES (:user_id)"),
                            {"user_id": self.current_user.id}
                        )
                        session.commit()

                    # Update the user type if needed
                    session.execute(
                        text("UPDATE users SET type = 'kivy_user' WHERE id = :user_id"),
                        {"user_id": self.current_user.id}
                    )
                    session.commit()

                    # Now fetch the KivyUser
                    self.current_user = session.query(KivyUser).filter(KivyUser.id == self.current_user.id).first()

                    if not isinstance(self.current_user, KivyUser):
                        raise ValueError("Failed to convert user to KivyUser")

                except Exception as e:
                    print(f"Error converting user to KivyUser: {e}")
                    raise ValueError("Current user is not a KivyUser instance and conversion failed")

            # Gather layout information from panels
            layout_info = {}
            for panel in self.get_all_panels():
                layout_info[panel.title] = {
                    'x': panel.x,
                    'y': panel.y,
                    'width': panel.width,
                    'height': panel.height,
                    'is_expanded': panel.is_expanded,
                }

            # Save to database via KivyUser
            self.current_user.save_layout(layout_name, layout_info)

            snackbar = MDSnackbar()
            snackbar.text = f"Layout '{layout_name}' saved successfully"
            snackbar.open()

            # Update layout spinner
            self.update_layout_spinner()

        except Exception as e:
            print(f"Error saving layout: {e}")
            import traceback
            traceback.print_exc()

            snackbar = MDSnackbar()
            snackbar.text = f"Error saving layout: {e}"
            snackbar.open()

    def update_layout_spinner(self):
        """Update the layout spinner with saved layout names from database"""
        if not hasattr(self, 'layout_spinner'):
            return

        try:
            layouts = {}

            if isinstance(self.current_user, KivyUser):
                # Get layouts from database
                layouts = self.current_user.get_all_layouts()

            elif hasattr(self, 'default_layouts'):
                # Use default layouts if user is not KivyUser but defaults are available
                layouts = self.default_layouts

            names = list(layouts.keys())
            if names:
                self.layout_spinner.values = names

                # Select "Default" layout by default if it exists
                if "Default" in names:
                    self.layout_spinner.text = "Default"
                else:
                    self.layout_spinner.text = "Select Layout"
            else:
                self.layout_spinner.values = []
                self.layout_spinner.text = "No saved layouts"

        except Exception as e:
            print(f"Error updating layout spinner: {e}")
            self.layout_spinner.values = []

    def load_asset_position(self, asset):
        """Load the position for a selected asset by first checking positions via asset number.
        If none are found, look up the AssetNumber by asset.number and then use the model id
        associated with that AssetNumber to find a matching Position.
        """
        try:
            print(f"LOADING POSITION FOR ASSET #{asset.number}, ID={asset.id}")
            logger.info(f"=========== LOADING POSITION FOR ASSET #{asset.number} ===========")
            logger.info(f"Asset ID: {asset.id}, Description: {asset.description or 'None'}")

            # Close the search results dialog, if it's open
            if hasattr(self, 'search_results_dialog'):
                print("Closing search results dialog")
                self.search_results_dialog.dismiss()

            session = self.data_service.session

            # 1. Look for a position where the asset_number_id matches asset.id
            print(f"Querying for position with asset_number_id = {asset.id}")
            position = session.query(Position).filter(Position.asset_number_id == asset.id).first()

            # 2. If no position is found, use the asset's number to search for matching AssetNumber records.
            if not position:
                print(f"No position found for Asset #{asset.number} using asset id")
                logger.warning(f"No position found for asset #{asset.number} via asset_number_id")

                # Use the asset number (string) to get a list of matching AssetNumber ids
                asset_ids = AssetNumber.get_ids_by_number(session, asset.number)
                if asset_ids:
                    # For simplicity, we pick the first asset_id that matched.
                    chosen_asset_id = asset_ids[0]
                    logger.info(f"Using AssetNumber id {chosen_asset_id} from search by number")

                    # Look up the model id using the chosen asset_number id
                    model_id = AssetNumber.get_model_id_by_asset_number_id(session, chosen_asset_id)
                    if model_id:
                        logger.info(f"Found model_id {model_id} from AssetNumber id {chosen_asset_id}")
                        # Search for a position with this model_id (if available)
                        position = session.query(Position).filter(Position.model_id == model_id).first()
                        if position:
                            logger.info(f"Found position ID {position.id} using model_id {model_id}")
                        else:
                            logger.warning(f"No position found with model_id {model_id}")
                            snackbar = MDSnackbar()
                            snackbar.text = f"No position found for Asset #{asset.number} or its model"
                            snackbar.open()
                            return
                    else:
                        logger.warning(f"Could not obtain model_id from AssetNumber id {chosen_asset_id}")
                        snackbar = MDSnackbar()
                        snackbar.text = f"No position found for Asset #{asset.number} (model lookup failed)"
                        snackbar.open()
                        return
                else:
                    logger.warning(f"No AssetNumber records found for asset number '{asset.number}'")
                    snackbar = MDSnackbar()
                    snackbar.text = f"No position found for Asset #{asset.number}"
                    snackbar.open()
                    return

            # If we got here, a position was found (either directly or via the fallback path)
            print(f"Found position ID: {position.id}")
            logger.info(f"Found position ID: {position.id}")
            logger.debug(f"Position details: Area={position.area_id}, Group={position.equipment_group_id}, "
                         f"Model={position.model_id}, Location={position.location_id}")

            # (Following sections update spinners and current state as in your existing code)
            model = session.query(Model).filter(Model.id == position.model_id).first()
            group = session.query(EquipmentGroup).filter(EquipmentGroup.id == position.equipment_group_id).first()
            area = session.query(Area).filter(Area.id == position.area_id).first()
            location = session.query(Location).filter(Location.id == position.location_id).first()

            logger.info(f"Position hierarchy: Area={area.name if area else 'None'}, "
                        f"Group={group.name if group else 'None'}, "
                        f"Model={model.name if model else 'None'}, "
                        f"Location={location.name if location else 'None'}")

            # Update the navigation spinners (areas, groups, models, locations) accordingly
            if area:
                self.areas_spinner.text = area.name
                areas = self.data_service.get_all_areas()
                area_obj = next((a for a in areas if a.name == area.name), None)
                if area_obj:
                    groups = self.data_service.get_equipment_groups_by_area(area_obj.id)
                    if groups:
                        self.groups_spinner.values = [g.name for g in groups]
            if group:
                self.groups_spinner.text = group.name
                groups = self.data_service.get_equipment_groups_by_area(area.id)
                group_obj = next((g for g in groups if g.name == group.name), None)
                if group_obj:
                    models = self.data_service.get_models_by_equipment_group(group_obj.id)
                    if models:
                        self.models_spinner.values = [m.name for m in models]
            if model:
                self.models_spinner.text = model.name
                models = self.data_service.get_models_by_equipment_group(group.id)
                model_obj = next((m for m in models if m.name == model.name), None)
                if model_obj:
                    locations = self.data_service.get_locations_by_model(model_obj.id)
                    if locations:
                        self.locations_spinner.values = [l.name for l in locations]
            if location:
                self.locations_spinner.text = location.name

            # Set current hierarchy IDs (for later filtering/updating)
            self.current_area_id = position.area_id
            self.current_group_id = position.equipment_group_id
            self.current_model_id = position.model_id
            self.current_location_id = position.location_id
            self.current_position_id = position.id

            # Optionally clear previous problem filters and refresh content panels
            if hasattr(self, 'current_problem_id'):
                self.current_problem_id = None
            self.refresh_panel_content_with_filters()

            # Final feedback to the user
            snackbar = MDSnackbar()
            snackbar.text = f"Loaded Asset #{asset.number} in {area.name if area else ''} / " \
                            f"{group.name if group else ''} / {model.name if model else ''}"
            snackbar.open()
            logger.info(f"=========== ASSET POSITION LOADING COMPLETE ===========")

        except Exception as e:
            logger.error(f"Error loading asset position: {e}")
            import traceback
            logger.error(traceback.format_exc())
            snackbar = MDSnackbar()
            snackbar.text = f"Error loading asset position: {e}"
            snackbar.open()

    def refresh_panel_content_with_filters(self):
        """Refresh all panel content based on current hierarchy filters or specific position"""
        try:
            logger.info("============== STARTING PANEL REFRESH ==============")
            logger.info("Getting panels for refresh")
            panels = self.get_all_panels()
            logger.info(f"Found {len(panels)} panels to refresh")
            position_ids = []

            # Log the current state
            logger.info("Current filter state:")
            area_id_str = f"{self.current_area_id}" if hasattr(self,
                                                               'current_area_id') and self.current_area_id else "Not set"
            group_id_str = f"{self.current_group_id}" if hasattr(self,
                                                                 'current_group_id') and self.current_group_id else "Not set"
            model_id_str = f"{self.current_model_id}" if hasattr(self,
                                                                 'current_model_id') and self.current_model_id else "Not set"
            location_id_str = f"{self.current_location_id}" if hasattr(self,
                                                                       'current_location_id') and self.current_location_id else "Not set"
            position_id_str = f"{self.current_position_id}" if hasattr(self,
                                                                       'current_position_id') and self.current_position_id else "Not set"

            logger.info(f"  Area ID: {area_id_str}")
            logger.info(f"  Group ID: {group_id_str}")
            logger.info(f"  Model ID: {model_id_str}")
            logger.info(f"  Location ID: {location_id_str}")
            logger.info(f"  Position ID: {position_id_str}")

            # If we have a specific position ID, prioritize that over filters
            if hasattr(self, 'current_position_id') and self.current_position_id != -1:
                logger.info(f"Using specific position ID: {self.current_position_id}")
                position_ids = [self.current_position_id]

                # Let's also log the hierarchy this position belongs to
                try:
                    position = self.data_service.session.query(Position).filter(
                        Position.id == self.current_position_id).first()
                    if position:
                        logger.info(
                            f"Position details: Area={position.area_id}, Group={position.equipment_group_id}, " +
                            f"Model={position.model_id}, Location={position.location_id}, Asset={position.asset_number_id}")
                    else:
                        logger.warning(f"Position ID {self.current_position_id} not found in database!")
                except Exception as e:
                    logger.error(f"Error fetching position details: {e}")
            else:
                # Otherwise, get positions matching the current filters
                logger.info("Using hierarchy filters to find positions")
                query = self.data_service.session.query(Position.id).distinct()

                filter_count = 0
                if self.current_area_id:
                    logger.info(f"Applying area filter: {self.current_area_id}")
                    query = query.filter(Position.area_id == self.current_area_id)
                    filter_count += 1

                if self.current_group_id:
                    logger.info(f"Applying group filter: {self.current_group_id}")
                    query = query.filter(Position.equipment_group_id == self.current_group_id)
                    filter_count += 1

                if self.current_model_id:
                    logger.info(f"Applying model filter: {self.current_model_id}")
                    query = query.filter(Position.model_id == self.current_model_id)
                    filter_count += 1

                if self.current_location_id:
                    logger.info(f"Applying location filter: {self.current_location_id}")
                    query = query.filter(Position.location_id == self.current_location_id)
                    filter_count += 1

                logger.info(f"Applied {filter_count} filters to position query")

                # Execute the query and get results
                try:
                    position_ids = [pos_id for pos_id, in query.all()]
                    logger.info(f"Found {len(position_ids)} positions through filters: {position_ids}")
                except Exception as e:
                    logger.error(f"Error executing position query: {e}")
                    position_ids = []

            if not position_ids:
                logger.warning("No positions found matching the current filters or position ID")
                # No positions found, clear all panels
                for panel in panels:
                    if not panel.children:
                        logger.debug(
                            f"Panel '{panel.title if hasattr(panel, 'title') else 'Unknown'}' has no children to clear")
                        continue

                    content_widget = panel.children[0]
                    logger.info(
                        f"Clearing panel: {panel.title if hasattr(panel, 'title') else 'Unknown'} - Widget type: {type(content_widget).__name__}")

                    # Clear all content widgets
                    if isinstance(content_widget, ProblemSolutionContent):
                        logger.debug("Clearing ProblemSolutionContent widget")
                        content_widget.problem_container.clear_widgets()
                        content_widget.solution_container.clear_widgets()
                        content_widget.task_container.clear_widgets()
                        content_widget.task_details.text = ""
                        content_widget.problem_container.add_widget(
                            OneLineListItem(text="No positions found for this selection")
                        )
                    elif isinstance(content_widget, PartsContent):
                        logger.debug("Clearing PartsContent widget")
                        content_widget.parts_container.clear_widgets()
                        content_widget.parts_container.add_widget(
                            OneLineListItem(text="No positions found for this selection")
                        )
                    elif isinstance(content_widget, DocumentsContent):
                        logger.debug("Clearing DocumentsContent widget")
                        content_widget.docs_container.clear_widgets()
                        content_widget.doc_content.text = ""
                        content_widget.button_container.clear_widgets()
                        content_widget.docs_container.add_widget(
                            OneLineListItem(text="No positions found for this selection")
                        )
                    elif isinstance(content_widget, ImagesContent):
                        logger.debug("Clearing ImagesContent widget")
                        content_widget.images_container.clear_widgets()
                        content_widget.image_preview.source = ''
                        content_widget.image_description.text = ""
                        content_widget.images_container.add_widget(
                            OneLineListItem(text="No positions found for this selection")
                        )
                    elif isinstance(content_widget, DrawingsContent):
                        logger.debug("Clearing DrawingsContent widget")
                        content_widget.drawings_container.clear_widgets()
                        content_widget.drawing_preview.source = ''
                        content_widget.details_layout.clear_widgets()
                        content_widget.drawings_container.add_widget(
                            OneLineListItem(text="No positions found for this selection")
                        )
                    else:
                        logger.warning(f"Unknown content widget type: {type(content_widget).__name__}")

                logger.info("Finished clearing all panels - returning from refresh")
                return

            # Explicitly reference ProblemSolutionContent panel
            logger.info("Looking for Problem/Solution panel")
            problem_solution_panel = next(
                (p for p in panels if hasattr(p, 'title') and p.title == "Problem / Solution"),
                None
            )

            if problem_solution_panel and problem_solution_panel.children:
                logger.info("Found Problem/Solution panel with children")
                problem_solution_content = problem_solution_panel.children[0]

                # Decision point: either get problems by specific position or by filters
                if hasattr(self, 'current_position_id') and self.current_position_id != -1:
                    # If we have a specific position, get problems for that position directly
                    logger.info(f"Getting problems for specific position ID: {self.current_position_id}")
                    try:
                        problems = self.data_service.get_problems_by_position(self.current_position_id)
                        logger.info(f"Retrieved {len(problems)} problems for position ID {self.current_position_id}")
                    except Exception as e:
                        logger.error(f"Error getting problems by position: {e}")
                        problems = []
                else:
                    # Otherwise use filters
                    logger.info("Getting problems by hierarchy filters")
                    try:
                        problems = self.data_service.get_problems_by_filters(
                            area_id=self.current_area_id,
                            equipment_group_id=self.current_group_id,
                            model_id=self.current_model_id,
                            location_id=self.current_location_id
                        )
                        logger.info(f"Retrieved {len(problems)} problems via filters")
                    except Exception as e:
                        logger.error(f"Error getting problems by filters: {e}")
                        problems = []

                logger.debug("Clearing problem solution content containers")
                problem_solution_content.problem_container.clear_widgets()
                problem_solution_content.solution_container.clear_widgets()
                problem_solution_content.task_container.clear_widgets()
                problem_solution_content.task_details.text = ""

                if not problems:
                    logger.info("No problems found, adding empty message")
                    problem_solution_content.problem_container.add_widget(
                        OneLineListItem(text="No problems found for this selection")
                    )
                else:
                    logger.info(f"Found {len(problems)} problems to display")
                    for i, problem in enumerate(problems):
                        logger.debug(f"Adding problem #{i + 1}: ID={problem.id}, Name='{problem.name}'")
                        from kivymd.uix.button import MDFlatButton

                        problem_item = MDFlatButton(
                            text=problem.name,
                            size_hint_y=None,
                            height=dp(60),
                            halign="left"
                        )
                        problem_item.problem = problem
                        # Explicit binding fixed here
                        problem_item.bind(
                            on_release=lambda btn, p=problem,
                                              content=problem_solution_content: content.show_solutions_for_problem(p)
                        )
                        problem_solution_content.problem_container.add_widget(problem_item)
            else:
                if not problem_solution_panel:
                    logger.error("Problem/Solution panel not found among the panels!")
                else:
                    logger.error("Problem/Solution panel found but has no children!")

            # Handle other content widgets (Parts, Documents, Images, Drawings)
            if position_ids:
                position_id = position_ids[0]  # Just use the first position if multiple
                logger.info(f"Using position ID {position_id} for panel content")

                # Query and log information about this position
                try:
                    position = self.data_service.session.query(Position).filter(Position.id == position_id).first()
                    if position:
                        logger.info(f"Selected position details:")
                        logger.info(f"  Area ID: {position.area_id}")
                        logger.info(f"  Group ID: {position.equipment_group_id}")
                        logger.info(f"  Model ID: {position.model_id}")
                        logger.info(f"  Location ID: {position.location_id}")

                        # Get more details about the asset if available
                        if position.asset_number_id:
                            try:
                                asset = self.data_service.session.query(AssetNumber).filter(
                                    AssetNumber.id == position.asset_number_id).first()
                                if asset:
                                    logger.info(f"  Asset Number: {asset.number}")
                                    logger.info(f"  Asset Description: {asset.description}")
                            except Exception as asset_err:
                                logger.error(f"Error getting asset details: {asset_err}")
                except Exception as pos_err:
                    logger.error(f"Error getting position details: {pos_err}")

                # Update each panel
                for panel in panels:
                    if not panel.children:
                        logger.debug(
                            f"Panel '{panel.title if hasattr(panel, 'title') else 'Unknown'}' has no children to update")
                        continue

                    content_widget = panel.children[0]
                    panel_name = panel.title if hasattr(panel, 'title') else "Unknown"
                    logger.info(f"Updating panel: {panel_name} - Widget type: {type(content_widget).__name__}")

                    try:
                        if isinstance(content_widget, PartsContent):
                            logger.debug(f"Updating PartsContent with position ID {position_id}")
                            content_widget.update_for_position(position_id)
                        elif isinstance(content_widget, DocumentsContent):
                            logger.debug(f"Updating DocumentsContent with position ID {position_id}")
                            content_widget.update_for_position(position_id)
                        elif isinstance(content_widget, ImagesContent):
                            logger.debug(f"Updating ImagesContent with position ID {position_id}")
                            content_widget.update_for_position(position_id)
                        elif isinstance(content_widget, DrawingsContent):
                            logger.debug(f"Updating DrawingsContent with position ID {position_id}")
                            content_widget.update_for_position(position_id)
                        else:
                            logger.debug(f"No update method for panel type: {type(content_widget).__name__}")
                    except Exception as panel_err:
                        logger.error(f"Error updating {panel_name} panel: {panel_err}")

            logger.info("============== FINISHED PANEL REFRESH ==============")

        except Exception as e:
            logger.error(f"Error refreshing panel content with filters: {e}")
            import traceback
            traceback.print_exc()
            snackbar = MDSnackbar()
            snackbar.text = f"Error refreshing data: {e}"
            snackbar.open()

    def on_asset_button_click(self, instance):
        """Callback for when an asset button is clicked"""
        asset = instance.asset_data
        print(f"Asset button clicked: #{asset.number}")
        self.load_asset_position(asset)

    def perform_asset_autocomplete(self, query):
        """
        Performs an autocomplete lookup using the AssetNumber.search_asset_numbers class
        method and shows a dropdown menu with suggestions.
        """
        # If the query is empty, dismiss any existing menu.
        if not query.strip():
            if hasattr(self, 'asset_autocomplete_menu') and self.asset_autocomplete_menu:
                self.asset_autocomplete_menu.dismiss()
                self.asset_autocomplete_menu = None
            return

        try:
            # Get suggestions from your backend.
            suggestions = AssetNumber.search_asset_numbers(self.data_service.session, query)
            # Build the list of menu items.
            menu_items = []
            for item in suggestions:
                # Make a copy of the dictionary so we capture it in the closure.
                current_item = dict(item)
                # (Optionally, you can add the text to the dictionary.)
                current_item["display_text"] = f"{current_item['number']} - {current_item['description'] or ''}"
                menu_items.append({
                    "viewclass": "OneLineListItem",
                    "text": current_item["display_text"],
                    # Capture the current_item explicitly in the lambda.
                    "on_release": lambda *args, current_item=current_item: self.on_asset_autocomplete_select(
                        current_item)
                })

            # If a previous menu exists, dismiss it.
            if hasattr(self, 'asset_autocomplete_menu') and self.asset_autocomplete_menu:
                self.asset_autocomplete_menu.dismiss()
                self.asset_autocomplete_menu = None

            # Create a new dropdown menu.
            self.asset_autocomplete_menu = MDDropdownMenu(
                caller=self.asset_search_field,
                items=menu_items,
                width_mult=4,
            )
            self.asset_autocomplete_menu.open()

        except Exception as e:
            print(f"Error during asset autocomplete: {e}")

    def on_asset_autocomplete_select(self, selected_item):
        """
        Called when a suggestion is selected.
        Updates the search field and triggers full search.
        """
        if not selected_item:
            print("DEBUG: on_asset_autocomplete_select called with None")
            return  # Guard: do nothing if selected_item is falsy

        # Use the captured "display_text" from the dictionary.
        text_value = selected_item.get("display_text", "")
        print("DEBUG: Selected item display_text:", text_value)
        if not text_value:
            print("DEBUG: No display_text found in selected item.")
            return

        # Extract the asset number (assuming format "ASSET_NUMBER - DESCRIPTION")
        asset_number_text = text_value.split(" - ")[0]
        print("DEBUG: Extracted asset number:", asset_number_text)
        self.asset_search_field.text = asset_number_text

        # Dismiss and remove the menu.
        if hasattr(self, 'asset_autocomplete_menu') and self.asset_autocomplete_menu:
            self.asset_autocomplete_menu.dismiss()
            self.asset_autocomplete_menu = None

        # Optionally trigger the full search (for example, as if the search button was clicked).
        self.on_search_button_click(None)

    def on_asset_search_text(self, instance, text):
        """
        Called whenever the asset search field text changes.
        Uses Clock.schedule_once to debounce the autocomplete search.
        """
        # Cancel any previously scheduled search
        if hasattr(self, '_asset_search_trigger'):
            self._asset_search_trigger.cancel()
        # Schedule a new search after 0.3 seconds.
        self._asset_search_trigger = Clock.schedule_once(lambda dt: self.perform_asset_autocomplete(text), 0.3)

    def on_clear_button_click(self, instance):
        """
        Clears the search field text, dismisses any autocomplete menu and search results dialog.
        """
        print("Clear button clicked; clearing search results.")
        # Clear the search field text.
        self.asset_search_field.text = ""

        # Dismiss the autocomplete menu if it exists.
        if hasattr(self, 'asset_autocomplete_menu') and self.asset_autocomplete_menu:
            self.asset_autocomplete_menu.dismiss()
            self.asset_autocomplete_menu = None

        # Dismiss the search results dialog if it is open.
        if hasattr(self, 'search_results_dialog') and self.search_results_dialog:
            self.search_results_dialog.dismiss()
            self.search_results_dialog = None

        # Optionally, you can also clear any other search-related state variables.
        print("Search box and results cleared.")

    def on_clear_position_click(self, instance):
        """
        Clears all set position filters and resets the navigation spinners and the current position state.
        """
        print("Clear position button clicked; clearing set position filters.")

        # Reset the spinners to their default values.
        self.areas_spinner.text = "Select an area"
        self.groups_spinner.text = "Select a group"
        self.models_spinner.text = "Select a model"
        self.locations_spinner.text = "Select a location"

        # Optionally clear the dropdown values as well.
        self.groups_spinner.values = []
        self.models_spinner.values = []
        self.locations_spinner.values = []

        # Reset the current set position state variables.
        self.current_area_id = None
        self.current_group_id = None
        self.current_model_id = None
        self.current_location_id = None
        self.current_position_id = -1

        # Refresh the panels to clear content related to the previous position.
        self.refresh_panel_content_with_filters()

        # Provide a snackbar or other user feedback.
        snackbar = MDSnackbar()
        snackbar.text = "Position filters cleared."
        snackbar.open()

    def load_layout(self):
        """Load the selected layout configuration from database"""
        if not hasattr(self, 'layout_spinner'):
            snackbar = MDSnackbar()
            snackbar.text = "Layout spinner not found"
            snackbar.open()
            return

        layout_name = self.layout_spinner.text
        if layout_name == "Select Layout" or layout_name == "No saved layouts":
            snackbar = MDSnackbar()
            snackbar.text = "Please select a layout to load"
            snackbar.open()
            return

        try:
            # Load layout from database via KivyUser
            if not isinstance(self.current_user, KivyUser):
                raise ValueError("Current user is not a KivyUser instance")

            layout_info = self.current_user.get_layout(layout_name)

            if not layout_info:
                snackbar = MDSnackbar()
                snackbar.text = f"Layout '{layout_name}' not found"
                snackbar.open()
                return

            # Apply the layout to panels
            panels = self.get_all_panels()
            for panel in panels:
                if panel.title in layout_info:
                    panel_config = layout_info[panel.title]
                    panel.pos = (panel_config['x'], panel_config['y'])
                    panel.size = (panel_config['width'], panel_config['height'])
                    if panel_config.get('is_expanded', False) != panel.is_expanded:
                        panel.toggle_expand()

            # Show success message
            snackbar = MDSnackbar()
            snackbar.text = f"Layout '{layout_name}' loaded successfully"
            snackbar.open()

        except Exception as e:
            print(f"Error loading layout: {e}")
            import traceback
            traceback.print_exc()

            snackbar = MDSnackbar()
            snackbar.text = f"Error loading layout: {e}"
            snackbar.open()

    def logout(self, instance):
        """Log out the current user and return to login screen"""
        self.current_user = None

        # Remove the main screen from the screen manager
        self.screen_manager.remove_widget(self.main_screen)

        # Reset the screen manager to the login screen
        self.screen_manager.current = "login_screen"

        # Clear sensitive data
        login_screen = self.screen_manager.get_screen("login_screen")
        login_screen.employee_id.text = ""
        login_screen.password.text = ""
        login_screen.error_label.text = ""

        # Show a snackbar message
        snackbar = MDSnackbar()
        snackbar.text = "Logged out successfully"
        snackbar.open()

    def delete_layout(self, layout_name=None):
        """Delete a layout from the database"""
        if not layout_name:
            layout_name = self.layout_spinner.text

        if layout_name == "Select Layout" or layout_name == "No saved layouts":
            snackbar = MDSnackbar()
            snackbar.text = "Please select a layout to delete"
            snackbar.open()
            return

        try:
            # Delete layout from database
            if not isinstance(self.current_user, KivyUser):
                raise ValueError("Current user is not a KivyUser instance")

            if self.current_user.delete_layout(layout_name):
                snackbar = MDSnackbar()
                snackbar.text = f"Layout '{layout_name}' deleted successfully"
                snackbar.open()

                # Update layout spinner
                self.update_layout_spinner()
            else:
                snackbar = MDSnackbar()
                snackbar.text = f"Layout '{layout_name}' not found"
                snackbar.open()

        except Exception as e:
            print(f"Error deleting layout: {e}")

            snackbar = MDSnackbar()
            snackbar.text = f"Error deleting layout: {e}"
            snackbar.open()

    def initialize_default_layouts(self):
        """Initialize default layouts if they don't exist"""
        default_layouts = {
            "Default": {
                "Problem / Solution": {"x": 10.0, "y": 10.0, "width": 940.0, "height": 929.0, "is_expanded": False},
                "Documents": {"x": 960.0, "y": 489.5, "width": 470.0, "height": 459.5, "is_expanded": False},
                "Parts": {"x": 1440.0, "y": 489.5, "width": 470.0, "height": 459.5, "is_expanded": False},
                "Images": {"x": 960.0, "y": 10.0, "width": 470.0, "height": 459.5, "is_expanded": False},
                "Drawings": {"x": 1440.0, "y": 10.0, "width": 470.0, "height": 459.5, "is_expanded": False}
            }
        }

        try:
            # If the user is a KivyUser, save these default layouts
            if isinstance(self.current_user, KivyUser):
                for layout_name, layout_data in default_layouts.items():
                    # Check if layout already exists
                    existing_layout = self.current_user.get_layout(layout_name)
                    if not existing_layout:
                        # Save the default layout
                        self.current_user.save_layout(layout_name, layout_data)
                        print(f"Created default layout: {layout_name}")

                # Update the layout spinner
                self.update_layout_spinner()

                # Apply default layout
                Clock.schedule_once(lambda dt: self.apply_default_layout(), 0.1)
            else:
                # If not a KivyUser, store the defaults for later use
                self.default_layouts = default_layouts

                # Apply default layout
                Clock.schedule_once(lambda dt: self.apply_default_layout(), 0.1)
        except Exception as e:
            print(f"Error initializing default layouts: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    MaintenanceTroubleshootingApp().run()