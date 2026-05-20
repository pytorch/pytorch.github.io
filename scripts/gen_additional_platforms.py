#!/usr/bin/env python3
"""
Generates additional platforms quick start module for 
https://pytorch.org/get-started/additional-platforms/ page.

This script reads all JSON files from _additional_platforms/ directory,
combines them, and generates the quick-start-additional-platforms.js file.
It also reads markdown files from _get_started/additional_platforms/ directory
and embeds them for dynamic content loading.

Usage:
    python3 scripts/gen_additional_platforms.py

The script will:
1. Read all JSON files from _additional_platforms/ directory
2. Read all MD files from _get_started/additional_platforms/ directory
3. Combine platform data into a single object (keyed by filename)
4. Replace template placeholders in _includes/quick-start-additional-platforms.js
5. Output the result to assets/quick-start-additional-platforms.js
"""

import json
from pathlib import Path
from typing import Dict, Any
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
import re

BASE_DIR = Path(__file__).parent.parent
ADDITIONAL_PLATFORM_DIR = BASE_DIR / "_additional_platforms"
MARKDOWN_DIR = BASE_DIR / "_get_started" / "additional_platforms"
INCLUDES_DIR = BASE_DIR / "_includes"
ASSETS_DIR = BASE_DIR / "assets"


def read_platform_json_files() -> Dict[str, Any]:
    """Read all JSON files from _additional_platforms directory."""
    platform_data = {}
    
    if not ADDITIONAL_PLATFORM_DIR.exists():
        print(f"Warning: {ADDITIONAL_PLATFORM_DIR} does not exist")
        return platform_data
    
    for json_file in ADDITIONAL_PLATFORM_DIR.glob("*.json"):
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


def convert_markdown_to_html(markdown_text: str) -> str:
    """Convert markdown to HTML with syntax highlighting.
    
    Uses Python markdown library with codehilite extension for syntax highlighting.
    This produces HTML similar to Jekyll's markdownify filter.
    """
    # Remove Jekyll/Kramdown special syntax (e.g., {:.no_toc}, {: #id})
    html_text = re.sub(r'\{:[^}]*\}', '', markdown_text)
    
    # Convert markdown to HTML with codehilite extension
    # Codehilite uses Pygments for syntax highlighting (similar to Rouge used by Jekyll)
    md = markdown.Markdown(extensions=[
        CodeHiliteExtension(
            css_class='highlight',
            guess_lang=False,
            linenums=False
        ),
        'fenced_code',
        'tables',
        'toc'
    ])
    
    html = md.convert(html_text)
    return html


def read_markdown_files() -> Dict[str, str]:
    """Read all markdown files from _get_started/additional_platforms directory
    and convert them to HTML with syntax highlighting."""
    html_content = {}
    
    if not MARKDOWN_DIR.exists():
        print(f"Warning: {MARKDOWN_DIR} does not exist")
        return html_content
    
    for md_file in MARKDOWN_DIR.glob("*.md"):
        try:
            content = md_file.read_text()
            # Use filename (without .md) as platform_id
            platform_id = md_file.stem
            # Convert markdown to HTML with syntax highlighting
            html = convert_markdown_to_html(content)
            html_content[platform_id] = html
            print(f"Loaded and converted markdown: {platform_id} from {md_file.name}")
        except Exception as e:
            print(f"Error reading {md_file.name}: {e}")
    
    return html_content


def read_template() -> str:
    """Read the JS template file."""
    template_path = INCLUDES_DIR / "quick-start-additional-platforms.js"
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    return template_path.read_text()


def generate_js_output(platform_data: Dict[str, Any], markdown_content: Dict[str, str]) -> str:
    """Generate the final JS file by replacing template placeholders."""
    template = read_template()
    
    # Replace placeholders
    template = template.replace("{{ platformData }}", json.dumps(platform_data, indent=2))
    # Now embedding pre-converted HTML content instead of raw markdown
    template = template.replace("{{ markdownContent }}", json.dumps(markdown_content, indent=2))
    
    return template


def write_output(content: str) -> None:
    """Write the generated JS to assets directory."""
    output_path = ASSETS_DIR / "quick-start-additional-platforms.js"
    output_path.write_text(content)
    print(f"Generated: {output_path}")


def main():
    """Main entry point."""
    print("Generating additional platforms quick start module...")
    
    # Read all platform JSON files
    platform_data = read_platform_json_files()
    
    if not platform_data:
        print("No platform data found. Creating empty output.")
        platform_data = {}
    
    # Read all markdown files
    markdown_content = read_markdown_files()
    
    if not markdown_content:
        print("No markdown content found.")
        markdown_content = {}
    
    # Generate JS output
    js_content = generate_js_output(platform_data, markdown_content)
    
    # Write to assets directory
    write_output(js_content)
    
    print("Done!")


if __name__ == "__main__":
    main()
