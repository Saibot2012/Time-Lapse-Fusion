from operator import truediv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import os

from PIL.ImageOps import scale

'Normal fusion equation'
def contrast(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(grayscale, cv.CV_32F)
    abs_laplacian = np.abs(laplacian)
    return(abs_laplacian)

def saturation(img):
    return np.std(img, axis=2)

def well_exposure(img):
    sigma = 0.9
    eqn = np.exp(-((img - 0.5) ** 2 / (2.0 * sigma ** 2)))
    eqn = np.prod(eqn, axis=2)
    return eqn

def weighted_map(img, wc=1, ws=1, we=1):
    C = contrast(img)
    S = saturation(img)
    E = well_exposure(img)

    C = (C - C.min()) / (C.max() - C.min() + 1e-12)
    S = (S - S.min()) / (S.max() - S.min() + 1e-12)
    E = (E - E.min()) / (E.max() - E.min() + 1e-12)

    W = (C ** wc) * (S ** ws) * (E ** we) + 1e-5
    return W


def normalise_weights(weights):
    weights = np.stack(weights, axis=0)
    total = np.sum(weights, axis=0, keepdims=True)
    return list(weights / (total + 1e-12))


def individual_pixel_weight(images):
    weights = []

    for img in images:
        weights.append(weighted_map(img))

    return weights


def multiply_weights_with_images(images, weights):
    weighted_img = []

    for img, W in zip(images, weights):
        W = W[:, :, None]
        weighted_img.append(img * W)

    return weighted_img


def summation(weighted_img):
    fusion = 0
    for img in weighted_img:
        fusion += img
    return fusion


"Improved equation"

def gaussian_pyramid(img, levels):
    G = [img]
    for i in range(levels):
        next_level = cv.pyrDown(G[i])
        G.append(next_level)
    return G

def get_weight_pyramids(images, levels):
    weight_pyramids = []
    for img in images:
        w = weighted_map(img)
        gp = gaussian_pyramid(w, levels)
        weight_pyramids.append(gp)


    return weight_pyramids


def normalise_weight_pyramids(weight_pyramids):
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
        next_level = cv.pyrUp(G[i + 1])
        next_level = next_level[:current_level.shape[0], :current_level.shape[1]]

        laplacian = current_level - next_level
        l.append(laplacian)

    l.append(G[-1])
    return l


def laplacian_result(laplacian_pyramid_output, individual_pixel_weights, levels):
    blended = []

    for i in range(levels + 1):
        level_array = np.zeros_like(laplacian_pyramid_output[0][i])

        for k in range(len(laplacian_pyramid_output)):
            fusion = laplacian_pyramid_output[k][i] * individual_pixel_weights[k][i][:, :, None]
            level_array += fusion

        blended.append(level_array)
        print(f"Level {i} mean weight: {np.mean(individual_pixel_weights[0][i])}")

    return blended


def reconstruction_laplacian_result(blended, levels):
    current = blended[-1]

    for i in range(levels - 1, -1, -1):
        construct = cv.pyrUp(current)
        construct = construct[:blended[i].shape[0], :blended[i].shape[1]]
        current = construct + blended[i]

    return current



def mask_for_pyramids(mask_list, levels):
    pyramids = []

    for mask in mask_list:
        gp = [mask]

        for i in range(levels):
            gp.append(cv.pyrDown(gp[i]))

        pyramids.append(gp)

    return pyramids


def apply_masks(weights, mask):
    new_weight = weights * (1 - 0.7 * mask)
    return new_weight


def compute_alpha(images):
    img = cv.cvtColor(images, cv.COLOR_BGR2GRAY) / 255.0
    G = np.abs(img[:, 1:] - img[:, :-1])

    counts, bin_edges = np.histogram(G.flatten(), bins=50, range=(0, G.max()))
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    y = counts / counts.sum()

    mask = y > 0
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return None

    slope, intercept = np.polyfit(x, np.log(y), 1)
    return -slope


def find_sharpest(images):
    results = {}
    for i, img in enumerate(images):
        alpha = compute_alpha(img)
        if alpha is not None:
            results[i] = alpha  # store index not image

    best_index = min(results, key=lambda k: results[k])
    return images[best_index]  # return the actual cv2 image


def anti_ghosting(images):
    threshold = 1
    mask_list = []
    ref_image = find_sharpest(images)
    ref_image = cv.cvtColor(ref_image, cv.COLOR_BGR2GRAY).astype(np.float32) / 255

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32) /255
        motion = np.abs(gray - ref_image) #Measures how much the pixel has changed

        mask = motion / (threshold + 1e-12) #how much each pixel differs from the reference image
        mask = np.clip(mask, 0, 1) #Clips to 0-1

        mask_list.append(mask.astype(np.float32))

    return mask_list




num_frames_needed = 50 #Cannot be equalled to one or else code breaks


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
    return frames

def laplacian_score(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()




def main():
    output_dir = "/home/hog/generated_images"
    os.makedirs(output_dir, exist_ok=True)

    "Garden pictures"
    img1 = cv.imread("pictures/test1.jpeg")
    img2 = cv.imread("pictures/test2.jpeg")
    img3 = cv.imread("pictures/test3.jpeg")
    img4 = cv.imread("pictures/test4.jpeg")
    img5 = cv.imread("pictures/test5.jpeg")
    img6 = cv.imread("pictures/test6.jpeg")
    img7 = cv.imread("pictures/test7.jpeg")
    garden_images = [img1, img2, img3, img4, img5, img6, img7]

    "Park pictures"
    img1 = cv.imread("pictures/park1.jpeg")
    img2 = cv.imread("pictures/park2.jpeg")
    img3 = cv.imread("pictures/park3.jpeg")
    img4 = cv.imread("pictures/park4.jpeg")
    img5 = cv.imread("pictures/park5.jpeg")
    img6 = cv.imread("pictures/park6.jpeg")
    img7 = cv.imread("pictures/park7.jpeg")
    park_images = [img1, img2, img3, img5, img6, img7]

    "hallway pictures"
    img1 = cv.imread("pictures/hallway1.jpeg")
    img2 = cv.imread("pictures/hallway2.jpeg")
    img3 = cv.imread("pictures/hallway3.jpeg")
    img4 = cv.imread("pictures/hallway4.jpeg")
    img5 = cv.imread("pictures/hallway5.jpeg")
    img6 = cv.imread("pictures/hallway6.jpeg")
    img7 = cv.imread("pictures/hallway7.jpeg")
    img8 = cv.imread("pictures/hallway8.jpeg")
    img9 = cv.imread("pictures/hallway9.jpeg")
    hallway_images = [img1, img2, img3, img4, img5, img6, img7, img8, img9]

    "main building"
    img1 = cv.imread("pictures/main_building1.jpeg")
    img2 = cv.imread("pictures/main_building2.jpeg")
    img3 = cv.imread("pictures/main_building3.jpeg")
    img4 = cv.imread("pictures/main_building4.jpeg")
    img5 = cv.imread("pictures/main_building5.jpeg")
    img6 = cv.imread("pictures/main_building6.jpeg")
    img7 = cv.imread("pictures/main_building7.jpeg")
    mainbuilding_images = [img1, img2, img3, img4, img5, img6, img7]

    "cosmos"
    img1 = cv.imread("pictures/cosmos1.jpeg")
    img2 = cv.imread("pictures/cosmos2.jpeg")
    img3 = cv.imread("pictures/cosmos3.jpeg")
    img4 = cv.imread("pictures/cosmos4.jpeg")
    img5 = cv.imread("pictures/cosmos5.jpeg")
    img6 = cv.imread("pictures/cosmos6.jpeg")
    cosmos_images = [img1, img2, img3, img4, img5, img6]

    "moving"


    "Waterfall"
    img1 = cv.imread("pictures/Waterfall.jpg")
    img2 = cv.imread("pictures/Waterfall_over.jpg")
    img3 = cv.imread("pictures/Waterfall_under.jpg")
    waterfall_images = [img1, img2, img3]

    "Landscape"
    img1 = cv.imread("pictures/landscape.jpg")
    img2 = cv.imread("pictures/landscape_over.jpg")
    img3 = cv.imread("pictures/landscape_under.jpg")
    landscape_images = [img1, img2, img3]

    "Venice"
    img1 = cv.imread("pictures/venice.jpg")
    img2 = cv.imread("pictures/venice_over.jpg")
    img3 = cv.imread("pictures/venice_under.jpg")
    venice_images = [img1, img2, img3]

    "person"
    img1 = cv.imread("pictures/262A3282.tif")
    img2 = cv.imread("pictures/262A3283.tif")
    img3 = cv.imread("pictures/262A3284.tif")
    person_images = [img1, img2, img3]

    "lighthouse"
    img1 = cv.imread("pictures/Lighthouse.jpg")
    img2 = cv.imread("pictures/Lighthouse_over.jpg")
    img3 = cv.imread("pictures/Lighthouse_under.jpg")
    lighthouse_images = [img1, img2, img3]

    "Video test"
    video_images = video("pictures/blueLakeSunset.mp4", 3, 4)

    levels = 5

    options = [
        "1. Garden",
        "2. Park",
        "3. Hallway",
        "4. Main building",
        "5. Cosmos",
        "6. Moving",
        "7. Waterfall",
        "8. Landscape",
        "9. Venice",
        "10. Video",
        "11. Person",
        "12. Lighthouse",
    ]

    for i in range(0, len(options), 3):
        col1 = options[i]
        col2 = options[i + 1] if i + 1 < len(options) else ""
        col3 = options[i + 2] if i + 2 < len(options) else ""
        print(f"{col1:<25}{col2:<25}{col3}")

    intro = int(input("\nWhich dataset to use? "))

    if intro == 1:
        images = garden_images[:levels]
    elif intro == 2:
        images = park_images[:levels]
    elif intro == 3:
        images = hallway_images[:levels]
    elif intro == 4:
        images = mainbuilding_images[:levels]
    elif intro == 5:
        images = cosmos_images[:levels]
    elif intro == 7:
        images = waterfall_images[:levels]
    elif intro == 8:
        images = landscape_images[:levels]
    elif intro == 9:
        images = venice_images[:levels]
    elif intro == 10:
        images = video_images[:num_frames_needed]
    elif intro == 11:
        images = person_images[:num_frames_needed]
    elif intro == 12:
        images = lighthouse_images[:num_frames_needed]

    target_shape = images[0].shape[:2]

    images = [
        cv.resize(img, (target_shape[1], target_shape[0]))
        for img in images
        if img is not None
    ]

    alignMtb = cv.createAlignMTB()
    alignMtb.process(images, images)  # modifies in place

    scores = [(i, laplacian_score(img)) for i, img in enumerate(images)]
    scores.sort(key=lambda x: x[1], reverse=True)  # higher = sharper

    print("Laplacian ranking (best to worst):")
    for i, s in scores:
        print(i, s)

    if len(images) <= 10:
        k = len(images)  # static HDR
        best_indices = [i for i, _ in scores[:len(images)]]
        images = [images[i] for i in best_indices]
    else:
        k = int(0.4 * len(images))
        step = len(images) // k
        images = [images[i * step] for i in range(k)]  # evenly spread frames





    for i, img in enumerate(images):
        print(i, img.shape)
    mask_list = anti_ghosting(images)
    images = [img.astype(np.float32) / 255.0 for img in images]
    mask_pyramid = mask_for_pyramids(mask_list, levels) #For anti ghosting since not implemented not used.
    individual_weights = individual_pixel_weight(images)

    normalised_weights = normalise_weights(individual_weights)
    multiply_weights = multiply_weights_with_images(images, normalised_weights)
    total = summation(multiply_weights)
    clipped_total = np.clip(total, 0, 1)

    weight_gaussian_pyramid = get_weight_pyramids(images, levels)
    weight_gaussian_pyramid = normalise_weight_pyramids(weight_gaussian_pyramid)
    for i in range(len(images)):
        for l in range(levels + 1):
            weight_gaussian_pyramid[i][l] *= (1  - mask_pyramid[i][l])

    laplacian_pyr = [laplacian_pyramid(img, levels) for img in images]

    result = laplacian_result(laplacian_pyr, weight_gaussian_pyramid, levels)
    reconstruction = reconstruction_laplacian_result(result, levels)

    reconstruction = np.clip(reconstruction, 0, 1)
    reconstruction = (reconstruction * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 2)

    plt.subplot(121)
    plt.imshow(cv.cvtColor(clipped_total, cv.COLOR_BGR2RGB))
    plt.title("Naive version")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv.cvtColor(reconstruction, cv.COLOR_BGR2RGB))
    plt.title("Improved version")
    plt.axis('off')


    if intro == 1:
        fig.savefig(f"{output_dir}/comparison.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 2:
        fig.savefig(f"{output_dir}/park.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 3:
        fig.savefig(f"{output_dir}/hallway.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 4:
        fig.savefig(f"{output_dir}/mainbuilding.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 5:
        fig.savefig(f"{output_dir}/cosmos.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 6:
        fig.savefig(f"{output_dir}/moving.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 7:
        fig.savefig(f"{output_dir}/Waterfall.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 8:
        fig.savefig(f"{output_dir}/Landscape.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 9:
        fig.savefig(f"{output_dir}/Venice.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 10:
        fig.savefig(f"{output_dir}/video.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 11:
        fig.savefig(f"{output_dir}/person.png", bbox_inches="tight", pad_inches=0, dpi=400)
    elif intro == 12:
        fig.savefig(f"{output_dir}/lighthouse.png", bbox_inches="tight", pad_inches=0, dpi=400)
    plt.show()


main()