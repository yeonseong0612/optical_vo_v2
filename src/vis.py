import os
import numpy as np
import cv2

def vis_stereo_detected_keypoints(
    img_l_u8, img_r_u8,
    kp_l, kp_r,
    save_path,
    radius=2,
    step=1,
    draw_index=False
):
    """
    같은 시점(t-1) 좌/우 이미지를 좌우로 붙이고,
    각각 검출된 특징점을 표시.

    img_l_u8, img_r_u8: uint8 gray (H,W) or BGR (H,W,3)
    kp_l, kp_r: (N,2), (M,2) float32 [x,y]
    """
    def to_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    L = to_bgr(img_l_u8)
    R = to_bgr(img_r_u8)

    H = max(L.shape[0], R.shape[0])
    W = L.shape[1] + R.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:L.shape[0], :L.shape[1]] = L
    canvas[:R.shape[0], L.shape[1]:] = R
    offx = L.shape[1]

    kp_l = np.asarray(kp_l, np.float32)
    kp_r = np.asarray(kp_r, np.float32)

    # Left keypoints (초록)
    for i in range(0, kp_l.shape[0], step):
        x, y = kp_l[i]
        pt = (int(round(x)), int(round(y)))
        cv2.circle(canvas, pt, radius, (0, 255, 0), -1)
        if draw_index:
            cv2.putText(canvas, str(i), (pt[0] + 2, pt[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

    # Right keypoints (노랑)
    for j in range(0, kp_r.shape[0], step):
        x, y = kp_r[j]
        pt = (int(round(x)) + offx, int(round(y)))
        cv2.circle(canvas, pt, radius, (0, 255, 255), -1)
        if draw_index:
            cv2.putText(canvas, str(j), (pt[0] + 2, pt[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return canvas

def vis_stereo_matches(
    img_l_u8, img_r_u8,
    kp_l, kp_r,
    matches,               # (K,2) int (i_left, j_right)
    save_path,
    radius=2,
    step=1,
    draw_index=False,
    max_draw=800
):
    """
    t-1에서 stereo_matching으로 나온 matches를 시각화.
    좌/우를 붙이고, 매칭된 점만 찍고 선으로 연결.

    img_l_u8, img_r_u8: uint8 gray or BGR
    kp_l, kp_r: (N,2), (M,2)
    matches: (K,2) int array
    """
    def to_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    L = to_bgr(img_l_u8)
    R = to_bgr(img_r_u8)

    H = max(L.shape[0], R.shape[0])
    W = L.shape[1] + R.shape[1]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:L.shape[0], :L.shape[1]] = L
    canvas[:R.shape[0], L.shape[1]:] = R
    offx = L.shape[1]

    kp_l = np.asarray(kp_l, np.float32)
    kp_r = np.asarray(kp_r, np.float32)
    matches = np.asarray(matches, np.int32)

    if matches.size == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, canvas)
        return canvas

    # 너무 많으면 샘플링
    K = matches.shape[0]
    if K > max_draw:
        step = max(step, int(np.ceil(K / max_draw)))

    for t in range(0, K, step):
        iL, jR = matches[t]
        xl, yl = kp_l[iL]
        xr, yr = kp_r[jR]

        ptL = (int(round(xl)), int(round(yl)))
        ptR = (int(round(xr)) + offx, int(round(yr)))

        cv2.circle(canvas, ptL, radius, (0, 255, 0), -1)       # left: green
        cv2.circle(canvas, ptR, radius, (0, 255, 0), -1)       # right: green
        cv2.line(canvas, ptL, ptR, (0, 0, 255), 1)            # match line: red

        if draw_index:
            cv2.putText(canvas, f"{iL}->{jR}", (ptL[0] + 2, ptL[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return canvas

def vis_flow_arrows(
    img_t_u8,
    kp_tm1, kp_t,
    save_path,
    max_draw=300,
    min_len=1.0,
    max_len=30.0,
    thickness=1,
):
    """
    한 프레임(img_t) 위에 optical flow 벡터(화살표)로 시각화.
    kp_tm1 -> kp_t 를 화살표로 그림. (두 이미지 붙이지 않음)
    """
    if img_t_u8.ndim == 2:
        canvas = cv2.cvtColor(img_t_u8, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img_t_u8.copy()

    kp_tm1 = np.asarray(kp_tm1, np.float32)
    kp_t   = np.asarray(kp_t,   np.float32)
    N = min(len(kp_tm1), len(kp_t))
    if N == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, canvas)
        return canvas

    # 너무 많으면 랜덤 샘플링
    if N > max_draw:
        idx = np.random.choice(N, size=max_draw, replace=False)
        kp_tm1 = kp_tm1[idx]
        kp_t   = kp_t[idx]
        N = max_draw

    flow = kp_t - kp_tm1
    lens = np.sqrt((flow[:,0]**2 + flow[:,1]**2))

    # 너무 짧거나 너무 긴 건 잘라서 안정화(시각화용)
    keep = lens >= min_len
    kp_tm1 = kp_tm1[keep]
    kp_t   = kp_t[keep]
    flow   = flow[keep]
    lens   = lens[keep]

    if len(kp_tm1) == 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, canvas)
        return canvas

    # 길이 cap
    scale = np.ones_like(lens)
    scale[lens > max_len] = max_len / lens[lens > max_len]
    tip = kp_tm1 + flow * scale[:, None]

    for p0, p1 in zip(kp_tm1, tip):
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))
        cv2.arrowedLine(canvas, (x0,y0), (x1,y1), (0,0,255), thickness, tipLength=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return canvas

import cv2
import numpy as np

def vis_stereo_track_pairs_vertical(
    img_l,
    img_r,
    kp_l,
    kp_r,
    save_path=None,
    max_draw=300,
    step=1,
    pt_color_l=(0, 255, 0),    # 위(Left) 점
    pt_color_r=(0, 255, 255),  # 아래(Right) 점
    line_color=(0, 255, 255),
    radius=2,
    thickness=1,
):
    """
    논문식 stereo(track) 결과를 상–하 배치로 시각화

    위: Left image
    아래: Right image
    """

    # --- BGR 변환 ---
    if img_l.ndim == 2:
        img_l_vis = cv2.cvtColor(img_l, cv2.COLOR_GRAY2BGR)
    else:
        img_l_vis = img_l.copy()

    if img_r.ndim == 2:
        img_r_vis = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
    else:
        img_r_vis = img_r.copy()

    H, W = img_l_vis.shape[:2]

    # --- 상하 concat ---
    canvas = np.vstack([img_l_vis, img_r_vis])

    N = min(len(kp_l), len(kp_r))
    draw_cnt = 0

    for i in range(0, N, step):
        if draw_cnt >= max_draw:
            break

        xL, yL = kp_l[i]
        xR, yR = kp_r[i]

        # NaN 체크
        if not (np.isfinite(xL) and np.isfinite(yL) and
                np.isfinite(xR) and np.isfinite(yR)):
            continue

        xL_i, yL_i = int(round(xL)), int(round(yL))
        xR_i, yR_i = int(round(xR)), int(round(yR))

        # 범위 체크
        if not (0 <= xL_i < W and 0 <= yL_i < H):
            continue
        if not (0 <= xR_i < W and 0 <= yR_i < H):
            continue

        # 아래쪽 이미지 y offset
        yR_i_off = yR_i + H

        # draw
        cv2.circle(canvas, (xL_i, yL_i), radius, pt_color_l, -1)
        cv2.circle(canvas, (xR_i, yR_i_off), radius, pt_color_r, -1)
        cv2.line(
            canvas,
            (xL_i, yL_i),
            (xR_i, yR_i_off),
            line_color,
            thickness
        )

        draw_cnt += 1

    if save_path is not None:
        cv2.imwrite(save_path, canvas)

    return canvas

def vis_cycle_4panel(
    img_l_tm1_u8, img_r_tm1_u8,
    img_l_t_u8,   img_r_t_u8,
    kp_l_tm1, kp_l_t,
    kp_r_tm1, kp_r_t_of,   # kp_r_tm1 == right_kp_tm1, kp_r_t_of == p1_r2
    kp_r_t_st,             # kp_r_t_st == kp_r_refined
    save_path,
    max_draw=300,
    radius=2,
    draw_cycle_link=True
):

    def to_bgr(img):
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img.copy()

    Lm1 = to_bgr(img_l_tm1_u8)
    Rm1 = to_bgr(img_r_tm1_u8)
    Lt  = to_bgr(img_l_t_u8)
    Rt  = to_bgr(img_r_t_u8)

    # 크기 통일 (가장 단순하게: 좌상 크기에 맞춤)
    H, W = Lm1.shape[:2]
    Rm1 = cv2.resize(Rm1, (W, H), interpolation=cv2.INTER_AREA)
    Lt  = cv2.resize(Lt,  (W, H), interpolation=cv2.INTER_AREA)
    Rt  = cv2.resize(Rt,  (W, H), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((2*H, 2*W, 3), dtype=np.uint8)
    canvas[0:H,   0:W]   = Lm1
    canvas[0:H,   W:2*W] = Rm1
    canvas[H:2*H, 0:W]   = Lt
    canvas[H:2*H, W:2*W] = Rt

    # 오프셋 (각 패널의 좌상단)
    off_Lm1 = (0, 0)
    off_Rm1 = (W, 0)
    off_Lt  = (0, H)
    off_Rt  = (W, H)

    def add_off(pt, off):
        x, y = pt
        return (int(round(x)) + off[0], int(round(y)) + off[1])

    # 유효 길이
    N = min(
        kp_l_tm1.shape[0], kp_l_t.shape[0],
        kp_r_tm1.shape[0], kp_r_t_of.shape[0], kp_r_t_st.shape[0],
        max_draw
    )

    # 그리기
    for i in range(N):
        xLm1, yLm1 = kp_l_tm1[i]
        xLt,  yLt  = kp_l_t[i]
        xRm1, yRm1 = kp_r_tm1[i]
        xRof, yRof = kp_r_t_of[i]
        xRst, yRst = kp_r_t_st[i]

        if not (np.isfinite([xLm1,yLm1,xLt,yLt,xRm1,yRm1,xRof,yRof,xRst,yRst]).all()):
            continue

        pLm1 = add_off((xLm1, yLm1), off_Lm1)
        pLt  = add_off((xLt,  yLt ), off_Lt)
        pRm1 = add_off((xRm1, yRm1), off_Rm1)
        pRof = add_off((xRof, yRof), off_Rt)
        pRst = add_off((xRst, yRst), off_Rt)

        # 점 표시
        # L_{t-1}, L_t
        cv2.circle(canvas, pLm1, radius, (0, 255, 0), -1)   # green
        cv2.circle(canvas, pLt,  radius, (0, 255, 0), -1)

        # R_{t-1}
        cv2.circle(canvas, pRm1, radius, (255, 255, 0), -1) # cyan-ish

        # R_t (flow) / R_t (stereo)
        cv2.circle(canvas, pRof, radius, (0, 0, 255), -1)   # red = flow
        cv2.circle(canvas, pRst, radius, (0, 255, 255), -1) # yellow = stereo

        # 경로 선
        # Left temporal: L_{t-1} -> L_t
        cv2.line(canvas, pLm1, pLt, (0, 255, 0), 1)

        # Stereo(t-1): L_{t-1} -> R_{t-1} (상단 가로로 연결)
        cv2.line(canvas, pLm1, pRm1, (255, 255, 0), 1)

        # Right temporal: R_{t-1} -> R_t(flow)
        cv2.line(canvas, pRm1, pRof, (0, 0, 255), 1)

        # Stereo(t): L_t -> R_t(stereo) (하단 가로로 연결)
        cv2.line(canvas, pLt, pRst, (0, 255, 255), 1)

        # Cycle check: R_t(stereo) <-> R_t(flow)
        if draw_cycle_link:
            cv2.line(canvas, pRst, pRof, (255, 255, 255), 1)

    # 패널 경계선
    cv2.line(canvas, (W, 0), (W, 2*H), (80, 80, 80), 1)
    cv2.line(canvas, (0, H), (2*W, H), (80, 80, 80), 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, canvas)
    return canvas
