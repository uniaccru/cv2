"""Keypoint-based planar object tracking (SIFT localization + Lucas-Kanade tracking).

The first frame of the video is the reference image of the object.

Pipeline:
  * Localization: keypoints of the current frame are matched with the reference
    keypoints (kNN + Lowe ratio test), a homography is estimated with RANSAC.
    It is used to start tracking and to re-anchor it periodically (no drift).
  * Tracking: points that lie on the object are followed from frame to frame with
    pyramidal Lucas-Kanade optical flow (forward-backward check). A homography
    reference -> frame is fitted to them with RANSAC. New points are added on the
    object when old ones disappear, so the object can be followed while it
    becomes tiny or turns away.
  * The object is declared lost only when both tracking and localization fail.

Usage:
    python tracker.py input.avi output.avi --label "Mona Lisa"
"""
import argparse
import time

import cv2
import numpy as np

LK_PARAMS = dict(winSize=(21, 21), maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def create_detector(name):
    name = name.lower()
    if name == "sift":
        return cv2.SIFT_create(nfeatures=2000), cv2.NORM_L2
    if name == "orb":
        return cv2.ORB_create(nfeatures=2000), cv2.NORM_HAMMING
    if name == "akaze":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    raise ValueError(f"Unknown detector: {name}")


def project(pts, H):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def quad_size(quad):
    return float(np.sqrt(max(cv2.contourArea(quad.reshape(4, 2).astype(np.float32)), 1.0)))


def is_valid_quad(quad, frame_shape, prev_quad=None):
    """Reject degenerate or implausible projections of the (rectangular) object."""
    pts = quad.reshape(4, 2).astype(np.float32)
    if not np.all(np.isfinite(pts)) or np.abs(pts).max() > 1e5:
        return False
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return False
    area = cv2.contourArea(pts)
    frame_area = frame_shape[0] * frame_shape[1]
    if not (2e-4 * frame_area <= area <= 1.2 * frame_area):
        return False
    # Shape of a projected rectangle: sane angles and opposite sides of similar length.
    edges = np.roll(pts, -1, axis=0) - pts
    lens = np.linalg.norm(edges, axis=1)
    if lens.min() < 1e-3:
        return False
    for a, b in ((0, 2), (1, 3)):
        if not (0.4 <= lens[a] / lens[b] <= 2.5):
            return False
    for i in range(4):
        u, v = -edges[i - 1], edges[i]
        cosang = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        if not (-0.82 <= cosang <= 0.82):  # angle between ~35 and ~145 degrees
            return False
    if prev_quad is not None:
        prev_area = cv2.contourArea(prev_quad.reshape(4, 2).astype(np.float32))
        if prev_area > 0 and not (0.5 <= area / prev_area <= 2.0):
            return False
    return True


def points_spread_ok(pts, quad, min_ratio=0.08):
    """Tracked points must cover a reasonable part of the object (no degenerate fit)."""
    if len(pts) < 4:
        return False
    hull = cv2.convexHull(np.float32(pts).reshape(-1, 1, 2))
    return cv2.contourArea(hull) >= min_ratio * cv2.contourArea(quad.reshape(4, 2).astype(np.float32))


def lk_track(prev, cur, pts):
    """Lucas-Kanade tracking with a forward-backward consistency check."""
    p0 = pts.reshape(-1, 1, 2).astype(np.float32)
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **LK_PARAMS)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **LK_PARAMS)
    fb = np.linalg.norm(p0r - p0, axis=2).ravel()
    h, w = cur.shape[:2]
    p1 = p1.reshape(-1, 2)
    inside = (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0) & inside
    return p1, ok


def build_reference(gray, detector, scales, mask=None, M=None):
    """Keypoints of the reference from several downscaled copies (mapped back).

    mask limits detection to the object; M maps first-frame coordinates into the
    reference (object) coordinate system.
    """
    pts_list, des_list = [], []
    for sc in scales:
        img, m = gray, mask
        if sc != 1.0:
            img = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if mask is not None:
                m = cv2.resize(mask, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        k, d = detector.detectAndCompute(img, m)
        if d is None or len(k) == 0:
            continue
        pts = np.float32([p.pt for p in k]) / sc
        if M is not None:
            pts = project(pts, M)
        pts_list.append(pts)
        des_list.append(d)
    if not des_list:
        raise SystemExit("No keypoints on the reference frame")
    return np.vstack(pts_list), np.vstack(des_list)


def localize(gray, detector, matcher, ref_pts, ref_des, ratio, ransac_thr, min_inliers):
    """Find the object in the frame by matching with the reference."""
    kp, des = detector.detectAndCompute(gray, None)
    if des is None or len(kp) < 8:
        return None
    knn = matcher.knnMatch(ref_des, des, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_inliers:
        return None
    src = ref_pts[[m.queryIdx for m in good]]
    dst = np.float32([kp[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2),
                                 cv2.RANSAC, ransac_thr)
    if H is None:
        return None
    m = mask.ravel().astype(bool)
    if m.sum() < min_inliers or m.sum() < 0.25 * len(good):
        return None
    return H, src[m], dst[m]


def recover(kf_gray, kf_H, kf_quad, gray, detector, matcher, ratio, min_inliers=8):
    """Re-acquire a lost object using the last frame where it was tracked.

    The object has roughly the same scale and appearance in that frame, so keypoints
    of the (upscaled) neighborhood of its last position are matched with the current
    frame. This works for small objects that cannot be matched with the large reference.
    """
    h, w = gray.shape[:2]
    size = quad_size(kf_quad)
    x0 = int(max(kf_quad[:, 0].min() - size, 0))
    y0 = int(max(kf_quad[:, 1].min() - size, 0))
    x1 = int(min(kf_quad[:, 0].max() + size, w))
    y1 = int(min(kf_quad[:, 1].max() + size, h))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    f = float(np.clip(200.0 / size, 1.0, 4.0))
    c1 = cv2.resize(kf_gray[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    c2 = cv2.resize(gray[y0:y1, x0:x1], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    mask = np.zeros(c1.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round((kf_quad - [x0, y0]) * f).astype(np.int32), 255)
    k1, d1 = detector.detectAndCompute(c1, mask)
    k2, d2 = detector.detectAndCompute(c2, None)
    if d1 is None or d2 is None or len(k1) < min_inliers or len(k2) < min_inliers:
        return None
    knn = matcher.knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < min_inliers:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    Hc, mk = cv2.findHomography(p1.reshape(-1, 1, 2), p2.reshape(-1, 1, 2), cv2.RANSAC, 3.0)
    if Hc is None:
        return None
    m = mk.ravel().astype(bool)
    if m.sum() < min_inliers or m.sum() < 0.25 * len(good):
        return None
    T = np.array([[f, 0, -x0 * f], [0, f, -y0 * f], [0, 0, 1]], np.float64)
    Tinv = np.linalg.inv(T)
    H = Tinv @ Hc @ T @ kf_H
    ref = project(project(p1[m], Tinv), np.linalg.inv(kf_H))
    cur = project(p2[m], Tinv)
    return H, ref, cur


def add_points(gray, quad, H, ref_pts, cur_pts, target):
    """Detect new trackable points on the object and add them (in reference coords)."""
    need = target - len(cur_pts)
    if need <= 0:
        return ref_pts, cur_pts
    mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    k = int(max(2, 0.06 * quad_size(quad)))
    mask = cv2.erode(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    for x, y in cur_pts:
        cv2.circle(mask, (int(x), int(y)), 3, 0, -1)
    new = cv2.goodFeaturesToTrack(gray, need, 0.005, 3, mask=mask, blockSize=5)
    if new is None:
        return ref_pts, cur_pts
    new = new.reshape(-1, 2)
    new_ref = project(new, np.linalg.inv(H))
    return np.vstack([ref_pts, new_ref]), np.vstack([cur_pts, new])


def draw_result(frame, quad, pts, label, color=(0, 255, 0)):
    k = max(1.0, frame.shape[1] / 640.0)  # scale drawing with the resolution
    q = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [q], True, color, int(round(2 * k)), cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), int(round(2 * k)), color, -1, cv2.LINE_AA)
    q = q.reshape(-1, 2)
    x, y = int(q[:, 0].min()), int(q[:, 1].min())
    fs, th_ = 0.7 * k, int(round(2 * k))
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_)
    x = max(x, 0)
    y = max(y, th + 10)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y + base - 4), color, -1)
    cv2.putText(frame, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (0, 0, 0), th_, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--label", default="object")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb", "akaze"])
    ap.add_argument("--ratio", type=float, default=0.8, help="Lowe ratio threshold")
    ap.add_argument("--ransac", type=float, default=4.0, help="RANSAC threshold for localization, px")
    ap.add_argument("--min-inliers", type=int, default=10, help="min inliers for localization")
    ap.add_argument("--min-track", type=int, default=8, help="min tracked points to keep tracking")
    ap.add_argument("--points", type=int, default=250, help="target number of tracked points")
    ap.add_argument("--relocalize-every", type=int, default=5,
                    help="re-anchor to the reference every N frames (0 = only after loss)")
    ap.add_argument("--recover-frames", type=int, default=90,
                    help="after a loss, try to recover from the last good frame for N frames")
    ap.add_argument("--scales", default="1,0.5,0.25,0.125,0.0625",
                    help="comma-separated scales of the reference pyramid")
    ap.add_argument("--roi", default=None,
                    help="object corners on the first frame as x1,y1,x2,y2,x3,y3,x4,y4 "
                         "(top-left, top-right, bottom-right, bottom-left); corners may lie "
                         "outside the frame. Without it the whole first frame is the object.")
    ap.add_argument("--log", default=None, help="write per-frame CSV log")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument("--no-points", action="store_true", help="do not draw tracked points")
    ap.add_argument("--show", action="store_true", help="show preview window")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    ok, ref = cap.read()
    if not ok:
        raise SystemExit("Cannot read the first frame")

    h, w = ref.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    detector, norm = create_detector(args.detector)
    matcher = cv2.BFMatcher(norm)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    # Object coordinate system ("reference"): either the whole first frame, or a
    # rectangle onto which the ROI quad of the first frame is mapped.
    if args.roi:
        q0 = np.float32([float(v) for v in args.roi.split(",")]).reshape(4, 2)
        rw = float(np.mean([np.linalg.norm(q0[1] - q0[0]), np.linalg.norm(q0[2] - q0[3])]))
        rh = float(np.mean([np.linalg.norm(q0[3] - q0[0]), np.linalg.norm(q0[2] - q0[1])]))
        ref_corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
        M = cv2.getPerspectiveTransform(q0, ref_corners)
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(roi_mask, np.round(q0).astype(np.int32), 255)
        k = int(max(3, 0.03 * quad_size(q0)))
        roi_mask = cv2.erode(roi_mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    else:
        # The reference frame is the object itself: its corners are the frame corners.
        ref_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        M, roi_mask = np.eye(3), None

    sift_pts, sift_des = build_reference(
        ref_gray, detector, [float(x) for x in args.scales.split(",")], roi_mask, M)
    print(f"Reference keypoints (all scales): {len(sift_pts)}")

    # Initial state: homography reference -> first frame, points taken from the first frame.
    H = np.linalg.inv(M)
    quad = project(ref_corners, H)
    p = cv2.goodFeaturesToTrack(ref_gray, args.points, 0.005, 5, mask=roi_mask, blockSize=5)
    cur_pts = p.reshape(-1, 2)
    ref_pts = project(cur_pts, M)

    first = ref.copy()
    draw_result(first, quad, [] if args.no_points else cur_pts, args.label)
    writer.write(first)

    log = open(args.log, "w") if args.log else None
    if log:
        log.write("frame,found,area,points,source\n")
    rng = np.random.default_rng(0)
    prev_gray = ref_gray
    kf_gray, kf_H, kf_quad, since_good = ref_gray, H.copy(), quad.copy(), 0
    total, found_frames, relocs, inlier_sum = 1, 1, 0, len(cur_pts)
    t0 = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        total += 1
        if args.max_frames and total > args.max_frames:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Tracking with optical flow.
        tracked = False
        source = "none"
        if H is not None and len(cur_pts) >= args.min_track:
            new, good = lk_track(prev_gray, gray, cur_pts)
            r, c = ref_pts[good], new[good]
            if len(r) >= args.min_track:
                thr = float(np.clip(0.01 * quad_size(quad), 1.0, 3.0))
                Ht, mask = cv2.findHomography(r.reshape(-1, 1, 2), c.reshape(-1, 1, 2),
                                              cv2.RANSAC, thr)
                if Ht is not None:
                    m = mask.ravel().astype(bool)
                    if m.sum() >= args.min_track:
                        q = project(ref_corners, Ht)
                        if is_valid_quad(q, frame.shape, quad) and \
                                points_spread_ok(c[m], q):
                            H, quad = Ht, q
                            ref_pts, cur_pts = r[m], c[m]
                            tracked = True
                            source = "lk"
        if not tracked:
            H = None
            ref_pts = np.empty((0, 2), np.float32)
            cur_pts = np.empty((0, 2), np.float32)

        # 2. Localization by reference keypoints: after a loss and periodically.
        periodic = args.relocalize_every > 0 and frame_idx % args.relocalize_every == 0
        if not tracked or periodic:
            res = localize(gray, detector, matcher, sift_pts, sift_des,
                           args.ratio, args.ransac, args.min_inliers)
            if res is not None:
                Hs, rs, cs = res
                q = project(ref_corners, Hs)
                if is_valid_quad(q, frame.shape) and points_spread_ok(cs, q) and \
                        (not tracked or len(rs) >= 15):
                    if tracked and len(cur_pts):
                        thr = float(np.clip(0.01 * quad_size(q), 1.0, 3.0))
                        keep = np.linalg.norm(project(ref_pts, Hs) - cur_pts, axis=1) < thr
                        ref_pts = np.vstack([ref_pts[keep], rs])
                        cur_pts = np.vstack([cur_pts[keep], cs])
                    else:
                        ref_pts, cur_pts = rs, cs
                    H, quad = Hs, q
                    tracked = True
                    source = "sift"
                    relocs += 1

        # 2b. Recovery from the last good frame (small objects, short losses).
        if not tracked and since_good <= args.recover_frames:
            res = recover(kf_gray, kf_H, kf_quad, gray, detector, matcher, args.ratio)
            if res is not None:
                Hr, rr, cr = res
                q = project(ref_corners, Hr)
                if is_valid_quad(q, frame.shape) and points_spread_ok(cr, q):
                    H, quad, ref_pts, cur_pts = Hr, q, rr, cr
                    tracked = True
                    source = "recover"
                    relocs += 1

        # 3. Replenish tracked points on the object (and cap their number).
        if tracked and len(cur_pts) > 2 * args.points:
            keep = rng.choice(len(cur_pts), 2 * args.points, replace=False)
            ref_pts, cur_pts = ref_pts[keep], cur_pts[keep]
        if tracked:
            kf_gray, kf_H, kf_quad, since_good = gray, H.copy(), quad.copy(), 0
            ref_pts, cur_pts = add_points(gray, quad, H, ref_pts, cur_pts, args.points)
            found_frames += 1
            inlier_sum += len(cur_pts)
            draw_result(frame, quad, [] if args.no_points else cur_pts, args.label)
        else:
            since_good += 1
            cv2.putText(frame, "lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if log:
            area = cv2.contourArea(quad.astype(np.float32)) if tracked else 0
            log.write(f"{frame_idx},{int(tracked)},{area:.0f},{len(cur_pts)},{source}\n")
        prev_gray = gray
        writer.write(frame)
        if args.show:
            cv2.imshow("tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    dt = time.time() - t0
    print(f"Frames: {total}, object found: {found_frames} ({100.0 * found_frames / total:.1f}%)")
    print(f"Re-localizations by reference keypoints: {relocs}")
    print(f"Mean tracked points: {inlier_sum / max(found_frames, 1):.1f}")
    print(f"Processing speed: {total / dt:.1f} fps")


if __name__ == "__main__":
    main()
