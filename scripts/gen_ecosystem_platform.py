#!/usr/bin/env python3
"""
Generates ecosystem platform quick start module for 
https://pytorch.org/get-started/ecosystem-platform/ page.

This script reads all JSON files from _ecosystem_platform/ directory,
combines them, and generates the quick-start-ecosystem-platform.js file.

Usage:
    python3 scripts/gen_ecosystem_platform.py

The script will:
1. Read all JSON files from _ecosystem_platform/ directory
2. Combine platform data into a single object (keyed by filename)
3. Replace template placeholders in _includes/quick-start-ecosystem-platform.js
4. Output the result to assets/quick-start-ecosystem-platform.js
"""

import json
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).parent.parent
ECOSYSTEM_PLATFORM_DIR = BASE_DIR / "_ecosystem_platform"
INCLUDES_DIR = BASE_DIR / "_includes"
ASSETS_DIR = BASE_DIR / "assets"


def read_platform_json_files() -> Dict[str, Any]:
    """Read all JSON files from _ecosystem_platform directory."""
    platform_data = {}
    
    if not ECOSYSTEM_PLATFORM_DIR.exists():
        print(f"Warning: {ECOSYSTEM_PLATFORM_DIR} does not exist")
        return platform_data
    
    for json_file in ECOSYSTEM_PLATFORM_DIR.glob("*.json"):
        try:
            content = json_file.read_text()
            data = json.loads(content)
            # Use filename (without .json) as platform_id
            platform_id = json_file.stem
            platform_data[platform_id] = data
            print(f"Loaded platform: {platform_id} from {json_file.name}")
        except json.JSONDecodeError as e:
            print(f"Error parsing {json_file.name}: {e}")
    
    return platform_data


def read_template() -> str:
    """Read the JS template file."""
    template_path = INCLUDES_DIR / "quick-start-ecosystem-platform.js"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text()


def generate_js_output(platform_data: Dict[str, Any]) -> str:
    """Generate the final JS file by replacing template placeholders."""
    template = read_template()
    
    # Replace placeholders
    template = template.replace("{{ platformData }}", json.dumps(platform_data, indent=2))
    
    return template


def write_output(content: str) -> None:
    """Write the generated JS to assets directory."""
    output_path = ASSETS_DIR / "quick-start-ecosystem-platform.js"
    output_path.write_text(content)
    print(f"Generated: {output_path}")


def main():
    """Main entry point."""
    print("Generating ecosystem platform quick start module...")
    
    # Read all platform JSON files
    platform_data = read_platform_json_files()
    
    if not platform_data:
        print("No platform data found. Creating empty output.")
        platform_data = {}
    
    # Generate JS output
    js_content = generate_js_output(platform_data)
    
    # Write to assets directory
    write_output(js_content)
    
    print("Done!")


if __name__ == "__main__":
    main()
