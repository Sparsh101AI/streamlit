import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Gum + Plaque + Cavity Detector", layout="centered")
st.title("Image Analysis")

uploaded = st.file_uploader("Upload a tooth image", type=["jpg", "jpeg", "png"])

tabs = st.tabs(["Plaque", "Cavity", "Gums"])

# -----------------------------
# HARD-CODED thresholds (no sliders)
# -----------------------------
# Plaque
PLAQUE_S_MIN = 60
PLAQUE_B_MIN = 140
PLAQUE_H_LOW = 10
PLAQUE_H_HIGH = 40
PLAQUE_L_MAX = 220

# Black spots / cavities
BLACK_V_MAX = 215
BLACK_S_MAX = 228
MIN_BLACK_AREA = 120

# Inflamed gums
INFLAMED_A_MIN = 176
INFLAMED_S_MIN = 70
MIN_INFLAMED_AREA = 150

# Optional toggles (not sliders)
with tabs[0]:
    show_plaque_overlay = st.checkbox("Show plaque overlay", value=True)
with tabs[1]:
    show_black_overlay = st.checkbox("Show black spot overlay", value=True)
with tabs[2]:
    show_inflamed_overlay = st.checkbox("Show inflamed overlay", value=True)


# -----------------------------
# Helpers for "good" messages
# -----------------------------
def has_any_pixels(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask)) > 0


def has_large_component(mask: np.ndarray, min_area: int) -> bool:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    return any(cv2.contourArea(c) >= min_area for c in contours)


def fill_black_holes_inside_white(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)
    inv = cv2.bitwise_not(padded)

    h, w = inv.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, ff_mask, (0, 0), 0)

    holes = inv
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def whiten_black_near_border(mask: np.ndarray, border_px: int = 8) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape

    band = np.zeros_like(m, dtype=np.uint8)
    band[:border_px, :] = 255
    band[-border_px:, :] = 255
    band[:, :border_px] = 255
    band[:, -border_px:] = 255

    black = (m == 0).astype(np.uint8) * 255
    black_in_band = cv2.bitwise_and(black, band)

    contours, _ = cv2.findContours(black_in_band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    to_whiten = np.zeros_like(m, dtype=np.uint8)

    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        seed_x, seed_y = x, y

        temp = black.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(temp, ff_mask, (seed_x, seed_y), 128)

        component = (temp == 128).astype(np.uint8) * 255
        to_whiten = cv2.bitwise_or(to_whiten, component)

    to_whiten = cv2.bitwise_and(to_whiten, band)
    m[to_whiten > 0] = 255
    return m


def detect_gums(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    red_hue = ((H <= 12) | (H >= 165))

    gum_mask = (
        red_hue &
        (S >= 60) &
        (V >= 60) &
        (A >= 150)
    ).astype(np.uint8) * 255

    gum_mask = cv2.morphologyEx(gum_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    gum_mask = cv2.morphologyEx(gum_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    gum_mask = fill_black_holes_inside_white(gum_mask)
    gum_mask = whiten_black_near_border(gum_mask, border_px=8)
    return gum_mask


def detect_plaque(
    rgb: np.ndarray,
    tooth_mask: np.ndarray,
    L_max: int,
    S_min: int,
    B_min: int,
    H_low: int,
    H_high: int
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    if H_low <= H_high:
        yellow_hue = (H >= H_low) & (H <= H_high)
    else:
        yellow_hue = (H >= H_low) | (H <= H_high)

    plaque = (
        yellow_hue &
        (S >= S_min) &
        (L <= L_max) &
        (B >= B_min)
    ).astype(np.uint8) * 255

    plaque = cv2.bitwise_and(plaque, tooth_mask)

    plaque = cv2.morphologyEx(plaque, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    plaque = cv2.morphologyEx(plaque, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    h, w = plaque.shape
    ignore_ratio = 0.30
    top_cutoff = int(h * ignore_ratio)

    vertical_mask = np.zeros_like(plaque, dtype=np.uint8)
    vertical_mask[top_cutoff:, :] = 255
    plaque = cv2.bitwise_and(plaque, vertical_mask)

    return plaque


def detect_black_spots(
    rgb: np.ndarray,
    tooth_mask: np.ndarray,
    V_max: int,
    S_max: int,
    min_area: int
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    black = ((V <= V_max) & (S <= S_max)).astype(np.uint8) * 255
    black = cv2.bitwise_and(black, tooth_mask)

    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(black)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(keep, [cnt], -1, 255, thickness=-1)

    return keep


def detect_inflamed_gums(
    rgb: np.ndarray,
    gum_mask: np.ndarray,
    A_min: int,
    S_min: int,
    min_area: int
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    inflamed = ((A >= A_min) & (S >= S_min)).astype(np.uint8) * 255
    inflamed = cv2.bitwise_and(inflamed, gum_mask)

    inflamed = cv2.morphologyEx(inflamed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    inflamed = cv2.morphologyEx(inflamed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(inflamed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(inflamed)

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(keep, [cnt], -1, 255, thickness=-1)

    return keep


def draw_boundaries_and_label(rgb: np.ndarray, mask: np.ndarray, label: str, color_bgr=(0, 120, 255)) -> np.ndarray:
    out = rgb.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out

    valid = [c for c in contours if cv2.contourArea(c) > 120]
    if not valid:
        return out

    for cnt in valid:
        cv2.drawContours(out, [cnt], -1, color_bgr, 3)

    largest = max(valid, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    cv2.putText(
        out,
        label,
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color_bgr,
        3,
        cv2.LINE_AA
    )
    return out


if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    rgb = np.array(img)

    gum_mask = detect_gums(rgb)

    gums_found = has_any_pixels(gum_mask)
    tooth_mask = cv2.bitwise_not(gum_mask) if gums_found else np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.uint8) * 255

    plaque_mask = detect_plaque(
        rgb, tooth_mask,
        PLAQUE_L_MAX, PLAQUE_S_MIN, PLAQUE_B_MIN,
        PLAQUE_H_LOW, PLAQUE_H_HIGH
    )

    black_spot_mask = detect_black_spots(
        rgb, tooth_mask,
        BLACK_V_MAX, BLACK_S_MAX, MIN_BLACK_AREA
    )

    inflamed_mask = detect_inflamed_gums(
        rgb, gum_mask if gums_found else np.zeros_like(tooth_mask),
        INFLAMED_A_MIN, INFLAMED_S_MIN, MIN_INFLAMED_AREA
    )

    st.subheader("Original")
    st.image(rgb, width="stretch")

    # -----------------------------
    # Plaque tab
    # -----------------------------
    with tabs[0]:
        plaque_ok = not has_large_component(plaque_mask, MIN_BLACK_AREA)  # reuse area threshold-ish
        if plaque_ok:
            st.success("✅ No plaque detected. Looks good!")
        else:
            st.warning("⚠️ Plaque-like regions detected.")

        plaque_mask_gray = np.where(plaque_mask > 0, 170, 0).astype(np.uint8)
        st.subheader("Plaque mask (gray)")
        st.image(plaque_mask_gray, width="stretch")

        if show_plaque_overlay and not plaque_ok:
            overlay = draw_boundaries_and_label(rgb, plaque_mask, "Plaque", color_bgr=(0, 120, 255))
            st.subheader("Overlay (plaque boundary highlighted)")
            st.image(overlay, width="stretch")

    # -----------------------------
    # Cavity tab
    # -----------------------------
    with tabs[1]:
        cavity_ok = not has_large_component(black_spot_mask, MIN_BLACK_AREA)
        if cavity_ok:
            st.success("✅ No cavities or dark spots detected. Looks good!")
        else:
            st.warning("⚠️ Dark spot regions detected (possible cavities).")

        black_mask_gray = np.where(black_spot_mask > 0, 170, 0).astype(np.uint8)
        st.subheader("Cavity Mask")
        st.image(black_mask_gray, width="stretch")

        if show_black_overlay and not cavity_ok:
            overlay = draw_boundaries_and_label(rgb, black_spot_mask, "Black spots", color_bgr=(255, 0, 0))
            st.subheader("Cavity Highlight")
            st.image(overlay, width="stretch")

    # -----------------------------
    # Gums tab
    # -----------------------------
    with tabs[2]:
        if not gums_found:
            st.success("✅ Gums not detected in this image (may be a tooth-only photo).")
        else:
            inflamed_ok = not has_large_component(inflamed_mask, MIN_INFLAMED_AREA)
            if inflamed_ok:
                st.success("✅ No inflamed gums detected. Looks good!")
            else:
                st.warning("⚠️ Inflamed gum-like regions detected.")

        inflamed_gray = np.where(inflamed_mask > 0, 170, 0).astype(np.uint8)
        st.subheader("Inflamed mask (gray)")
        st.image(inflamed_gray, width="stretch")

        if show_inflamed_overlay and gums_found:
            # only show overlay when there's actually something to show
            if has_large_component(inflamed_mask, MIN_INFLAMED_AREA):
                overlay = draw_boundaries_and_label(rgb, inflamed_mask, "Inflamed", color_bgr=(0, 0, 255))
                st.subheader("Overlay (inflamed boundary highlighted)")
                st.image(overlay, width="stretch")

else:
    st.info("Upload a tooth image to see results.")
