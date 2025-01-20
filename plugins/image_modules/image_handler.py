# plugins/image_modules/image_handler.py
from plugins.image_modules import CLIPModelHandler, NoImageModel
from modules.emtacdb.emtacdb_fts import Session, load_image_model_config_from_db

class ImageHandler:
    def __init__(self):
        self.model_handlers = {
            "clip": CLIPModelHandler(),
            "no_model": NoImageModel()
        }
        self.Session = Session
        self.current_model = load_image_model_config_from_db()

    def allowed_file(self, filename, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].allowed_file(filename)

    def preprocess_image(self, image, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].preprocess_image(image)

    def get_image_embedding(self, image, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].get_image_embedding(image)

    def is_valid_image(self, image, model_name=None):
        model_name = model_name or self.current_model
        return self.model_handlers[model_name].is_valid_image(image)

    def store_image_metadata(self, session, title, description, file_path, embedding, model_name=None):
        model_name = model_name or self.current_model
        self.model_handlers[model_name].store_image_metadata(session, title, description, file_path, embedding, model_name)
