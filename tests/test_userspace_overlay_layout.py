from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USERSPACE_PANEL = ROOT / "ragtime" / "frontend" / "src" / "components" / "UserSpacePanel.tsx"
PREVIEW_COMPONENT = ROOT / "ragtime" / "frontend" / "src" / "components" / "UserSpaceArtifactPreview.tsx"
COMPONENTS_CSS = ROOT / "ragtime" / "frontend" / "src" / "styles" / "components.css"


class UserSpaceOverlayLayoutTests(unittest.TestCase):
    def test_status_overlay_is_scoped_to_single_right_pane_section(self) -> None:
        panel = USERSPACE_PANEL.read_text(encoding="utf-8")

        self.assertEqual(panel.count("className={`userspace-preview-section"), 1)
        section_index = panel.index("className={`userspace-preview-section")
        overlay_mount_index = panel.index("{rightPaneCollapsed ? null : renderUserspaceOverlay()}", section_index)
        tab_branch_index = panel.index("{activeRightTab === 'preview' ? (")

        self.assertGreater(overlay_mount_index, section_index)
        self.assertLess(overlay_mount_index, tab_branch_index)
        self.assertIn(
            "{rightPaneCollapsed ? renderUserspaceOverlay(' userspace-status-overlay-root') : null}",
            panel,
        )
        self.assertNotIn("previewNotice={previewNotice}", panel)

    def test_preview_component_does_not_render_separate_message_banners(self) -> None:
        preview = PREVIEW_COMPONENT.read_text(encoding="utf-8")

        self.assertNotIn("userspace-preview-exec-notice", preview)
        self.assertNotIn("userspace-preview-exec-error", preview)

    def test_status_overlay_uses_preview_section_anchor_styles(self) -> None:
        css = COMPONENTS_CSS.read_text(encoding="utf-8")

        preview_section = re.search(r"\.userspace-preview-section \{(?P<body>.*?)\n\}", css, re.S)
        self.assertIsNotNone(preview_section)
        assert preview_section is not None
        self.assertIn("position: relative;", preview_section.group("body"))

        overlay = re.search(r"\.userspace-status-overlay \{(?P<body>.*?)\n\}", css, re.S)
        self.assertIsNotNone(overlay)
        assert overlay is not None
        overlay_body = overlay.group("body")
        self.assertIn("top: var(--space-sm);", overlay_body)
        self.assertIn("right: var(--space-sm);", overlay_body)
        self.assertNotIn("calc(44px", overlay_body)

        self.assertIn(".userspace-status-overlay-root", css)
        self.assertIn(".userspace-status-overlay-item.userspace-warning", css)
        self.assertIn(".userspace-status-overlay-item.userspace-success", css)


if __name__ == "__main__":
    unittest.main()
