#!/usr/bin/env python3
"""
PyCharm TODO Pattern Automation Script
Automatically adds TODO patterns to PyCharm by modifying the config files
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import re


def parse_markdown_file(markdown_file):
    """Parse the markdown file to extract TODO patterns"""
    patterns = []

    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for the summary table at the end
        table_match = re.search(r'### Summary Table.*?\n((?:\|.*?\n)+)', content, re.DOTALL)
        if not table_match:
            print("❌ Could not find Summary Table in markdown")
            return patterns

        table_lines = table_match.group(1).strip().split('\n')

        # Color mapping based on your exact markdown
        color_map = {
            'Teal': '#008080',
            'Red': '#D32F2F',
            'Orange': '#FF9800',
            'Grey': '#757575',
            'Blue': '#1976D2',
            'Purple': '#9C27B0',
            'Green': '#388E3C',
            'Orange-Red': '#FF5722',
            'Sky Blue': '#0288D1'
        }

        for line in table_lines:
            if not line.startswith('|') or 'Pattern' in line or '---' in line:
                continue

            cols = [col.strip() for col in line.split('|')[1:-1]]  # Remove empty first/last
            if len(cols) < 3:
                continue

            pattern_name = cols[0]
            color_name = cols[1]
            effects = cols[2]

            if not pattern_name or pattern_name == 'Pattern':
                continue

            # Handle escaped underscores in pattern names
            clean_pattern_name = pattern_name.replace('\\_', '_')

            hex_color = color_map.get(color_name, '#666666')

            pattern_data = {
                'name': clean_pattern_name,
                'pattern': f'\\b({re.escape(clean_pattern_name)})\\b.*',
                'color': hex_color,
                'bold': 'Bold' in effects,
                'italic': 'Italic' in effects,
                'underlined': 'Underline' in effects or 'UL' in effects
            }

            patterns.append(pattern_data)
            print(f"✓ Parsed: {clean_pattern_name} ({hex_color})")

        return patterns

    except Exception as e:
        print(f"❌ Error parsing markdown: {e}")
        return []


def find_pycharm_config():
    """Find PyCharm configuration directory"""
    home = Path.home()

    # Try different OS locations
    possible_paths = [
        home / "AppData" / "Roaming" / "JetBrains",  # Windows
        home / "Library" / "Application Support" / "JetBrains",  # macOS
        home / ".config" / "JetBrains"  # Linux
    ]

    for base_path in possible_paths:
        if base_path.exists():
            pycharm_dirs = [d for d in base_path.iterdir() if d.is_dir() and "PyCharm" in d.name]
            if pycharm_dirs:
                return pycharm_dirs

    return []


def update_todo_config(config_file, patterns):
    """Update the PyCharm TODO configuration file"""
    try:
        # Parse existing config if it exists
        if config_file.exists():
            tree = ET.parse(config_file)
            root = tree.getroot()
        else:
            root = ET.Element('application')
            tree = ET.ElementTree(root)

        # Remove existing TodoConfiguration component
        for component in root.findall('component[@name="TodoConfiguration"]'):
            root.remove(component)

        # Create new TodoConfiguration component
        todo_component = ET.SubElement(root, 'component', name='TodoConfiguration')

        # Add default TODO pattern
        default_pattern = ET.SubElement(todo_component, 'pattern')
        default_pattern.set('pattern', r'\btodo\b.*')
        default_pattern.set('attribs', '{"FOREGROUND":"0x0000ff","FONT_TYPE":"2","ERROR_STRIPE_COLOR":"0x0000ff"}')

        # Add custom patterns
        for pattern in patterns:
            font_type = 0
            if pattern['bold']: font_type += 1
            if pattern['italic']: font_type += 2
            if pattern['underlined']: font_type += 4

            color_hex = pattern['color'].replace('#', '0x')

            attribs = {
                'FOREGROUND': color_hex,
                'FONT_TYPE': str(font_type),
                'ERROR_STRIPE_COLOR': color_hex
            }

            attribs_str = '{' + ','.join([f'"{k}":"{v}"' for k, v in attribs.items()]) + '}'

            pattern_elem = ET.SubElement(todo_component, 'pattern')
            pattern_elem.set('pattern', pattern['pattern'])
            pattern_elem.set('attribs', attribs_str)
            pattern_elem.set('case-sensitive', 'true')

        # Write the file
        ET.indent(tree, space="  ")
        with open(config_file, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            tree.write(f, encoding='UTF-8')

        return True

    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False


def main():
    print("🚀 PyCharm TODO Pattern Automation")
    print("=" * 35)

    # Get markdown file path
    markdown_file = input("Enter markdown file path: ").strip().strip('"')

    if not Path(markdown_file).exists():
        print(f"❌ Markdown file not found: {markdown_file}")
        return

    # Parse patterns from markdown
    print(f"\n📖 Parsing patterns from: {Path(markdown_file).name}")
    patterns = parse_markdown_file(markdown_file)

    if not patterns:
        print("❌ No patterns found in markdown file")
        return

    print(f"✓ Found {len(patterns)} patterns")

    # Find PyCharm config directories
    print("\n🔍 Looking for PyCharm configuration...")
    pycharm_dirs = find_pycharm_config()

    if not pycharm_dirs:
        print("❌ PyCharm configuration not found")
        print("Make sure PyCharm is installed and has been run at least once")
        return

    # Select PyCharm version if multiple found
    if len(pycharm_dirs) > 1:
        print("Multiple PyCharm versions found:")
        for i, path in enumerate(pycharm_dirs):
            print(f"  {i + 1}. {path.name}")

        try:
            choice = int(input(f"Select version (1-{len(pycharm_dirs)}): ")) - 1
            selected_dir = pycharm_dirs[choice]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return
    else:
        selected_dir = pycharm_dirs[0]

    print(f"✓ Using: {selected_dir.name}")

    # Prepare config file path
    options_dir = selected_dir / "options"
    options_dir.mkdir(exist_ok=True)

    config_file = options_dir / "editor.xml"

    # Backup existing config
    if config_file.exists():
        backup_file = config_file.with_suffix('.backup')
        shutil.copy2(config_file, backup_file)
        print(f"✓ Backup created: {backup_file.name}")

    # Update configuration
    print(f"\n🔧 Updating TODO configuration...")
    success = update_todo_config(config_file, patterns)

    if success:
        print(f"✅ Configuration updated successfully!")
        print(f"📁 Config file: {config_file}")
        print("\n🔄 Restart PyCharm to see the new TODO patterns")

        print(f"\n🎯 Added {len(patterns)} patterns:")
        for pattern in patterns:
            effects = []
            if pattern['bold']: effects.append('bold')
            if pattern['italic']: effects.append('italic')
            if pattern['underlined']: effects.append('underlined')
            effect_str = '+'.join(effects) if effects else 'normal'
            print(f"  • {pattern['name']}: {pattern['color']} ({effect_str})")
    else:
        print("❌ Failed to update configuration")


if __name__ == "__main__":
    main()