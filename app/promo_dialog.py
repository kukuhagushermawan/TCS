"""PT Terramitra Citra Persada promotion popup shown every time Terra View opens."""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QPointF, QRectF, QTimer, QUrl, Qt
from PyQt6.QtGui import (
    QColor,
    QDesktopServices,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from .resources import resource_path

TERRAMITRA_URL = "https://www.terramitra.com"

TERRAMITRA_LOGO_CANDIDATES = (
    ("assets", "logos", "tcp_logo.png"),
)
HEADER_IMAGE_CANDIDATES = (
    ("assets", "opening", "popup_promo.png"),
)

DIALOG_WIDTH = 480
# The banner image is scaled to DIALOG_WIDTH (aspect ratio preserved) and then
# cropped down to this height, keeping the top (logos/title/description/most
# of the photo) and dropping only the bottom feature-badge strip - this is
# what keeps the popup landscape and compact instead of as tall as the full
# banner. 230px = ~80% of the scaled banner height, measured against the
# source image's own text bands so the description paragraph (ending ~77%)
# never gets cropped mid-sentence while the badge row (starting ~84%) does.
HEADER_IMAGE_HEIGHT = 230

# TCP's own brand colours (sampled from assets/opening/popup_promo.png), used
# instead of the main app's blue - this popup is deliberately branded as
# Terramitra's own material, not a screen of Terra View itself.
GREEN = "#15803D"
GREEN_DARK = "#0F6B31"
NAVY = "#0B1F33"
MUTED = "#64748B"


def _find_asset(candidates):
    for parts in candidates:
        candidate = resource_path(*parts)
        if candidate.exists():
            return candidate
    return None


class _LineIcon(QWidget):
    """Small flat, single-colour outline icon (phone/fax/chat/mail) used next
    to each contact line. Drawn with QPainter instead of an emoji character,
    so it stays a quiet monochrome pictogram rather than a multi-colour
    cartoon glyph."""

    def __init__(self, kind: str, size: int = 15, color_hex: str = GREEN, parent=None) -> None:
        super().__init__(parent)
        self._kind = kind
        self._color = QColor(color_hex)
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt method
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(self._color, 1.3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        w, h = float(self.width()), float(self.height())
        m = 2.0

        if self._kind == "phone":
            rect = QRectF(m + w * 0.16, m, w * 0.58, h - 2 * m)
            painter.drawRoundedRect(rect, 2.4, 2.4)
            cx = rect.center().x()
            painter.drawLine(QPointF(cx - 2.0, rect.bottom() - 3.2), QPointF(cx + 2.0, rect.bottom() - 3.2))
        elif self._kind == "fax":
            body = QRectF(m, h * 0.34, w - 2 * m, h * 0.44)
            painter.drawRoundedRect(body, 1.4, 1.4)
            painter.drawLine(QPointF(m + 1.5, h * 0.34), QPointF(w - m - 1.5, h * 0.34))
            tray = QRectF(w * 0.32, h - m - h * 0.16, w * 0.36, h * 0.16)
            painter.drawRect(tray)
        elif self._kind == "chat":
            bubble = QRectF(m, m, w - 2 * m, h * 0.62)
            painter.drawRoundedRect(bubble, 3.2, 3.2)
            path = QPainterPath()
            path.moveTo(w * 0.30, bubble.bottom() - 0.5)
            path.lineTo(w * 0.30, h - m)
            path.lineTo(w * 0.50, bubble.bottom() - 0.5)
            painter.drawPath(path)
        elif self._kind == "mail":
            rect = QRectF(m, h * 0.24, w - 2 * m, h * 0.52)
            painter.drawRoundedRect(rect, 1.4, 1.4)
            painter.drawLine(rect.topLeft(), rect.center())
            painter.drawLine(rect.topRight(), rect.center())

        painter.end()


class _WebsiteLabel(QLabel):
    """A clickable label - clicking anywhere on it fires on_click().

    A plain QLabel never fires anything on mouse press by itself (and even a
    real <a> anchor only reacts to the exact link span, not the surrounding
    text/padding), so opening the URL needs an explicit press handler on the
    whole label rather than relying on rich-text link clicks.
    """

    def __init__(self, text: str, on_click) -> None:
        super().__init__(text)
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt method
        self._on_click()


class TerramitraPromotionDialog(QDialog):
    """Non-blocking promotion popup for PT Terramitra Citra Persada.

    Shown every time Terra View opens (no permanent dismissal) - it never
    blocks the main window from being used. Any mouse click anywhere - on the
    popup itself or on the main window behind it - dismisses it, not just the
    Close button.

    Layout: the header is assets/opening/popup_promo.png pasted in as-is (not
    redrawn, just cropped shorter to keep the popup compact and landscape),
    followed by a contact section with a watermark logo and the action
    buttons - styled in Terramitra's own green/navy brand rather than the
    main app's palette, since this is the reseller's own promotional material.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PT Terramitra Citra Persada")
        icon_path = _find_asset(TERRAMITRA_LOGO_CANDIDATES)
        if icon_path is not None:
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setFixedWidth(DIALOG_WIDTH)
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setObjectName("PromoRoot")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header_image())
        outer.addWidget(self._divider())
        outer.addWidget(self._build_contact_section())
        outer.addLayout(self._build_buttons())

        self._apply_style()

        # Per-widget click-through tricks (WA_TransparentForMouseEvents,
        # per-child event filters) turned out unreliable - a click on any
        # label or on the app behind the popup would not close it. Watching
        # every mouse press application-wide is the only mechanism that is
        # guaranteed to see a click regardless of which widget it lands on,
        # inside the popup or in the main window underneath it.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # ---------- sections ----------
    def _build_header_image(self) -> QWidget:
        """The header is the supplied banner image, pasted in as-is - not
        redesigned, no text overlay - scaled to the dialog's width and then
        cropped down to HEADER_IMAGE_HEIGHT (keeping the top: logos, title,
        most of the photo) so the banner's own 5:3 aspect ratio doesn't force
        the whole popup to be tall."""
        label = QLabel()
        label.setObjectName("PromoHeaderImage")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_path = _find_asset(HEADER_IMAGE_CANDIDATES)
        if image_path is not None:
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                scaled = pixmap.scaledToWidth(DIALOG_WIDTH, Qt.TransformationMode.SmoothTransformation)
                cropped = scaled.copy(0, 0, scaled.width(), min(HEADER_IMAGE_HEIGHT, scaled.height()))
                label.setPixmap(cropped)
        return label

    def _divider(self) -> QWidget:
        line = QFrame()
        line.setObjectName("PromoDivider")
        line.setFixedHeight(1)
        return line

    def _build_contact_section(self) -> QWidget:
        section = QFrame()
        section.setObjectName("ContactSection")
        row = QHBoxLayout(section)
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(5)

        heading_row = QHBoxLayout()
        heading_row.setSpacing(7)
        heading_row.addWidget(_LineIcon("phone", size=15, color_hex=GREEN))
        heading = QLabel("HUBUNGI KAMI")
        heading.setObjectName("ContactHeading")
        heading_row.addWidget(heading)
        heading_row.addStretch()
        left.addLayout(heading_row)
        left.addSpacing(2)

        for kind, text in (
            ("phone", "(021) 2279 2937"),
            ("fax", "(021) 779 5539"),
            ("chat", "+62 851-1781-6507"),
            ("mail", "sales@terramitra.com; admin@terramitra.com"),
        ):
            contact_row = QHBoxLayout()
            contact_row.setSpacing(8)
            contact_row.addWidget(_LineIcon(kind, size=12, color_hex=MUTED), 0, Qt.AlignmentFlag.AlignVCenter)
            value = QLabel(text)
            value.setObjectName("ContactRow")
            value.setWordWrap(True)
            contact_row.addWidget(value, 1)
            left.addLayout(contact_row)

        row.addLayout(left, 3)

        watermark = QLabel()
        watermark.setObjectName("ContactWatermark")
        logo_path = _find_asset(TERRAMITRA_LOGO_CANDIDATES)
        if logo_path is not None:
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                watermark.setPixmap(pixmap.scaledToHeight(48, Qt.TransformationMode.SmoothTransformation))
        opacity = QGraphicsOpacityEffect(watermark)
        opacity.setOpacity(0.16)
        watermark.setGraphicsEffect(opacity)
        row.addWidget(watermark, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        return section

    def _build_buttons(self) -> QHBoxLayout:
        buttons = QHBoxLayout()
        buttons.setContentsMargins(20, 0, 20, 14)
        buttons.setSpacing(10)
        buttons.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setObjectName("PromoSecondaryButton")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close)
        btn_visit = QPushButton("Visit Website")
        btn_visit.setObjectName("PromoPrimaryButton")
        btn_visit.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_visit.clicked.connect(self._open_website)
        buttons.addWidget(btn_close)
        buttons.addWidget(btn_visit)
        return buttons

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QDialog#PromoRoot {{ background: #FFFFFF; }}
            QLabel#PromoHeaderImage {{ background: #FFFFFF; }}
            QFrame#PromoDivider {{ background: #E5E7EB; border: none; }}
            QFrame#ContactSection {{ background: #F8FAF9; }}
            QLabel#ContactHeading {{
                font-weight: 800; color: {GREEN}; font-size: 9.5pt; letter-spacing: 0.4px;
                background: transparent;
            }}
            QLabel#ContactRow {{ color: #374151; font-size: 8.5pt; background: transparent; }}
            QLabel#ContactWatermark {{ background: transparent; }}
            QPushButton#PromoPrimaryButton {{
                background: {GREEN}; color: white; border: none; padding: 8px 18px;
                border-radius: 4px; font-weight: 700; font-size: 9pt;
            }}
            QPushButton#PromoPrimaryButton:hover {{ background: {GREEN_DARK}; }}
            QPushButton#PromoPrimaryButton:pressed {{ background: {NAVY}; }}
            QPushButton#PromoSecondaryButton {{
                background: #FFFFFF; color: #374151; border: 1px solid #CBD5E1; padding: 8px 18px;
                border-radius: 4px; font-weight: 600; font-size: 9pt;
            }}
            QPushButton#PromoSecondaryButton:hover {{ background: #F1F5F9; }}
            """
        )

    # ---------- behaviour ----------
    def eventFilter(self, watched, event) -> bool:
        # Watch release, not press: QPushButton.clicked fires on release, so
        # scheduling the close on press used to tear the popup (and its
        # buttons) down before Qt ever delivered the release that would have
        # triggered "Visit Website" - the click was silently lost. Deferring
        # via a 0ms timer still lets that same release finish being
        # dispatched (and clicked() fire) before the popup actually closes.
        if event.type() == QEvent.Type.MouseButtonRelease:
            QTimer.singleShot(0, self._close_if_open)
        return False

    def _close_if_open(self) -> None:
        try:
            if self.isVisible():
                self.close()
        except RuntimeError:
            pass  # underlying C++ object already destroyed

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)

    def _open_website(self) -> None:
        QDesktopServices.openUrl(QUrl(TERRAMITRA_URL))
