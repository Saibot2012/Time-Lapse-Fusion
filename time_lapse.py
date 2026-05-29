import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
mask = None
drawing = False
mode = 1  # 1 for timelapse, 0 for static. can switch later


'time lapse + temporal image fusion equation'
def contrast(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(grayscale, cv.CV_32F)
    abs_laplacian = np.abs(laplacian)
    return(abs_laplacian)

def saturation(img):
    S = np.std(img, axis = 2)
    return S


def well_exposure(img):
    sigma = 0.2
    eqn = np.exp(-((img - 0.5) ** 2 / (2.0 * sigma ** 2)))
    eqn = np.prod(eqn, axis = 2)
    return(eqn)

def decay_constant(img):
    sigma_min = 7
    sigma_max = 15
    time_decay_value = sigma_min + (saturation(img) * (sigma_max - sigma_min))
    return time_decay_value

def time_decay(k, t, sigma):
    time_decay_value = np.exp(-((t - k) ** 2) / (2 * sigma ** 2))
    return time_decay_value

w = 5
TD_alpha = 20
def temporal_distinctness(frames, k, w):
    neighbor_frames = frames[max(0, k - w):min(len(frames), k + w + 1)]
    mu = np.mean(neighbor_frames, axis=0)
    diff = np.abs(frames[k] - mu)
    td = np.max(diff, axis=2)
    final_eqn = np.exp(np.clip(TD_alpha * td, 0, 5))
    return final_eqn




def weighted_map(img, k, t, frames):
    C = contrast(img)
    S = saturation(img)
    E = well_exposure(img)
    sigma = decay_constant(img)
    Time_w = time_decay(k, t, sigma)
    temporal_imaging = temporal_distinctness(frames, k, w)

    temporal_imaging = temporal_imaging / (temporal_imaging.max() + 1e-12)
    C = (C - C.min()) / (C.max() - C.min() + 1e-12) #Fix -1
    S = (S - S.min()) / (S.max() - S.min() + 1e-12)
    E = (E - E.min()) / (E.max() - E.min() + 1e-12)
    return C * S * E * Time_w


def normalise_weights(weights): #Fix 0
    weights = np.stack(weights, axis=0)
    total = np.sum(weights, axis=0, keepdims=True)
    return list(weights / (total + 1e-12))



def gaussian_pyramid(img, levels):
    G = [img]
    for i in range(levels):
        next_level = cv.pyrDown(G[i])
        G.append(next_level)
    return G





def normalise_weight_pyramids(weight_pyramids): #fix 1
    num_images = len(weight_pyramids)
    levels = len(weight_pyramids[0])

    for l in range(levels):
        stack = np.array([weight_pyramids[i][l] for i in range(num_images)])
        total = np.sum(stack, axis=0, keepdims=True)
        stack = stack / (total + 1e-12)

        for i in range(num_images):
            weight_pyramids[i][l] = stack[i]

    return weight_pyramids


def laplacian_pyramid(img, levels):
    l = []
    G = gaussian_pyramid(img, levels)

    for i in range(levels):
        current_level = G[i]
        next_level = cv.pyrUp(G[i+1])
        next_level = next_level[:current_level.shape[0], :current_level.shape[1]] #It crops the upsampled image so it matches the exact size of the current pyramid level.
        laplacian = current_level - next_level
        l.append(laplacian)

    l.append(G[-1])
    return l


def laplacian_result(laplacian_pyramid_output, individual_pixel_weights, levels):
    blended = []

    for i in range(levels+1):
        level_array = np.zeros_like(laplacian_pyramid_output[0][i])

        for k in range(len(laplacian_pyramid_output)):
            fusion = laplacian_pyramid_output[k][i] * individual_pixel_weights[k][i][:, :, None] #fix 2
            level_array += fusion

        blended.append(level_array)

    return blended


def reconstruction_laplacian_result(blended, levels):
    current = blended[-1]

    for i in range(levels - 1, -1, -1):
        construct = cv.pyrUp(current)
        construct = construct[:blended[i].shape[0], :blended[i].shape[1]] #Fix 3
        current = construct + blended[i]

    return current

num_frames_needed = 50
def video(video_name, start_time, end_time):
    cap = cv.VideoCapture(video_name)
    frames = []

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    FPS = int(cap.get(cv.CAP_PROP_FPS))

    if end_time is None:
        end_time = total_frames/ FPS
    start_frame = int(start_time * FPS)
    end_frame = int(end_time * FPS)

    step = (end_frame - start_frame) / (num_frames_needed - 1)

    for i in range(num_frames_needed):
        frame_index = int(start_frame + i * step)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)

        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames, FPS



def compute_all_weights(images, frames):
    t = len(images) // 2  # temporal center

    weights = []
    for k, img in enumerate(images):
        weight = weighted_map(img, k, t, frames)
        weights.append(weight)

    return weights

def compute_alpha_frame(frame):
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    G = np.abs(img[:, 1:] - img[:, :-1])

    counts, bin_edges = np.histogram(G.flatten(), bins=50, range=(0, 0.1))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = counts > 0
    x = x[mask]
    counts = counts[mask]

    y = counts / counts.sum()

    slope, intercept = np.polyfit(x, np.log(y), 1)

    return -slope


def main():
        output_dir = "/home/hog/generated_images"
        os.makedirs(output_dir, exist_ok=True)
        global mask

        "Video test"
        video_images, FPS = video("pictures/blueLakeSunset.mp4", 0, None)
        video_images2, FPS2 = video("pictures/timelapse2.mp4", 0, None)


        levels = 3

        options = [
            "1. Blue Lake Sunset",
            "2. Time Lapse Video 2"
        ]

        for i in range(0, len(options), 3):
            col1 = options[i]
            col2 = options[i + 1] if i + 1 < len(options) else ""
            col3 = options[i + 2] if i + 2 < len(options) else ""
            print(f"{col1:<25}{col2:<25}{col3}")

        intro = int(input("\nWhich dataset to use? "))

        if intro == 1:
            images, fps = video_images, FPS
        elif intro == 2:
            images, fps = video_images2, FPS2

        results = {}
        images = [img.astype(np.float32) / 255.0 for img in images]
        for i, img in enumerate(images):
            print(i, img.shape)
            alpha = compute_alpha_frame(img)
            results[f"frame_{i:03d}"] = alpha
            print(f"frame_{i:03d}: Alpha = {alpha:.2f}")

        "Print results"
        # Find the blurriest image (max alpha)
        blurriest = max(results, key=lambda k: results[k])
        print(f'\nBlurriest image: {blurriest} with Alpha = {results[blurriest]:.2f}')

        # Find the sharpest image (min alpha)
        sharpest = min(results, key=lambda k: results[k])
        print(f'Sharpest image: {sharpest} with Alpha = {results[sharpest]:.2f}')

        sharp_index = int(sharpest.split("_")[1])
        ref_frame = images[sharp_index]

        cv.destroyAllWindows()

        weights = compute_all_weights(images, images)
        weights = normalise_weights(weights)
        weight_pyramids = []

        for weight in weights:
            weight_pyramids.append(gaussian_pyramid(weight, levels))

        weight_pyramids = normalise_weight_pyramids(weight_pyramids)

        laplacian_pyr = [laplacian_pyramid(img, levels) for img in images]

        result = laplacian_result(laplacian_pyr, weight_pyramids, levels)
        reconstruction = reconstruction_laplacian_result(result, levels)


        final = np.clip(reconstruction, 0, 1)
        final = (final * 255).astype(np.uint8)
        fig, ax = plt.subplots(1, 2)

        plt.subplot(121)
        plt.imshow(cv.cvtColor(ref_frame, cv.COLOR_BGR2RGB))
        plt.title("Original version")
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv.cvtColor(final, cv.COLOR_BGR2RGB))
        plt.title("Time lapse version")
        plt.axis('off')

        if intro == 1:
            fig.savefig(f"{output_dir}/time_lapse_fusion.png", bbox_inches="tight", pad_inches=0, dpi=400)
        if intro == 2:
            fig.savefig(f"{output_dir}/time_lapse_fusion2.png", bbox_inches="tight", pad_inches=0, dpi=400)
        plt.show()

main()