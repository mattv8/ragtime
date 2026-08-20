import re
import unittest
from pathlib import Path

from ragtime.userspace.html_templates import render_share_unlock_prompt_html


class ShareUnlockThemeTests(unittest.TestCase):
    def test_render_injects_default_theme_pack_and_stable_asset(self) -> None:
        html = render_share_unlock_prompt_html(
            title='Unlock shared workspace',
            form_action='/shared/demo/authorize',
            subtitle='Workspace title',
            owner_label='Owner name',
            next_target='/shared/demo',
            default_theme_pack='vscode',
        )

        self.assertIn('href="/assets/share-theme.css"', html)
        self.assertIn('localStorage.getItem("ragtime-theme-pack")', html)
        self.assertIn('localStorage.getItem("ragtime-theme")', html)
        self.assertIn("['default', 'vscode', 'serif']", html)
        self.assertIn('"vscode"', html)

    def test_template_preserves_escaping_and_hidden_next_input(self) -> None:
        rendered = render_share_unlock_prompt_html(
            title='Unlock <workspace>',
            form_action='/shared/demo/authorize?x="1"',
            subtitle='Shared by <owner>',
            owner_label='Owner & Co',
            error='Bad <password>',
            next_target='/shared/demo?next="1"',
            default_theme_pack='serif',
        )

        self.assertIn('action="/shared/demo/authorize?x=&quot;1&quot;"', rendered)
        self.assertIn('name="next" value="/shared/demo?next=&quot;1&quot;"', rendered)
        self.assertIn('Unlock &lt;workspace&gt;', rendered)
        self.assertIn('Shared by &lt;owner&gt;', rendered)
        self.assertIn('Owner &amp; Co', rendered)
        self.assertIn('Bad &lt;password&gt;', rendered)
        self.assertNotIn('Bad <password>', rendered)

    def test_main_render_path_passes_default_theme_pack(self) -> None:
        main_source = (Path(__file__).resolve().parents[1] / 'ragtime' / 'main.py').read_text(encoding='utf-8')

        self.assertIn('default_theme_pack = await _get_share_unlock_default_theme_pack()', main_source)
        self.assertIn('default_theme_pack=default_theme_pack', main_source)
        self.assertIn('await _render_share_password_prompt(', main_source)
        self.assertIn('await _render_share_token_password_prompt(', main_source)

    def test_share_theme_entry_uses_shared_token_sources_without_duplicate_palette(self) -> None:
        frontend_dir = Path(__file__).resolve().parents[1] / 'ragtime' / 'frontend'
        css = (frontend_dir / 'src' / 'styles' / 'share-theme-entry.css').read_text(encoding='utf-8')
        template = (
            Path(__file__).resolve().parents[1]
            / 'ragtime'
            / 'userspace'
            / 'templates'
            / 'share_unlock_prompt.tsx'
        ).read_text(encoding='utf-8')

        self.assertIn("@import './fonts.css';", css)
        self.assertIn("@import './theme.css';", css)
        self.assertIn("@import './themes/vscode.css';", css)
        self.assertIn("@import './themes/serif.css';", css)
        self.assertIsNone(re.search(r"#[0-9a-fA-F]{3,8}", css))
        self.assertNotIn('#0f172a', template)
        self.assertNotIn('#111827', template)
