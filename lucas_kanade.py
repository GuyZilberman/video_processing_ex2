import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata


# FILL IN YOUR ID
ID1 = 308339274
ID2 = 212235246


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]

    for level_index in range(num_levels):
        # convolve the PYRAMID_FILTER with the image from the previous level
        filtered_level = signal.convolve2d(pyramid[level_index], PYRAMID_FILTER, boundary='symm', mode='same')
        # decimate the convolution result using indexing - pick every second entry of the result
        decimated_filtered_level = filtered_level[::2, ::2]
        # append the result
        pyramid.append(decimated_filtered_level)

    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    # Calculate Ix and Iy by convolving I2 with the appropriate filters
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, boundary='symm', mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
    It = I2.astype(int) - I1.astype(int)

    # Start from all-zeros du and dv (each one) of size I1.shape
    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    h, w = I1.shape
    # Loop over all pixels in the image (ignoring boundary pixels)
    half_window_size = window_size // 2
    for i in range(half_window_size, h - half_window_size):
        for j in range(half_window_size, w - half_window_size):
            # Extract the spatial and temporal gradient windows around the current pixel
            Ix_Window = Ix[i - half_window_size: i + half_window_size + 1, j - half_window_size : j + half_window_size + 1]
            Iy_Window = Iy[i - half_window_size: i + half_window_size + 1, j - half_window_size : j + half_window_size + 1]
            It_Window = It[i - half_window_size: i + half_window_size + 1, j - half_window_size: j + half_window_size + 1]

            # Reshape the gradient windows into vectors
            Ix_Vector = Ix_Window.reshape(-1)
            Iy_Vector = Iy_Window.reshape(-1)
            It_Vector = It_Window.reshape(-1)

            # Construct the A matrix and the b vector
            A = np.stack((Ix_Vector, Iy_Vector), axis=1)
            b = -It_Vector

            try:
                # Calculate the least squares solution, which is (u,v)
                least_squares_solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                # Extract du and dv from the least squares solution
                du[i, j] = least_squares_solution[0]
                dv[i, j] = least_squares_solution[1]
            except:
                # When the solution does not converge, keep this pixel's (u, v) as zero.
                pass
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    # Resize u and v to the shape of the input image
    u = cv2.resize(u, (image.shape[1], image.shape[0]))
    v = cv2.resize(v, (image.shape[1], image.shape[0]))

    # Get the u and v factors, and normalize according to them
    u_factor = image.shape[1] / u.shape[1]
    v_factor = image.shape[0] / v.shape[0]
    u *= u_factor
    v *= v_factor

    # Define the grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.flatten(), yy.flatten()]).T

    # Define the points to interpolate with coordinates (x+u, y+v)
    interp_points = points + np.vstack([u.flatten(), v.flatten()]).T

    # Interpolate the image at the interpolated points
    image_warp = griddata(points, image.flatten(), interp_points, method='linear', fill_value=np.nan)

    # Reshape the warped image from a flattened shape to the shape of the input image
    image_warp = image_warp.reshape(image.shape)

    # Fill the np.nan holes with the original image values
    image_warp[np.isnan(image_warp)] = image[np.isnan(image_warp)]

    return image_warp.astype(np.uint8)



def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    # Looping over every level in the image pyramid starting from the smallest image
    for level in range(num_levels, -1, -1):
        # Warp I2 from that level according to the current u and v
        I2_warp = warp_image(pyarmid_I2[level], u, v)
        # Repeat for num_iterations
        for iteration in range(max_iter):
            # Perform a Lucas Kanade Step with the I1 decimated image of the
            # current pyramid level and the current I2_warp to get the new I2_warp
            du, dv = lucas_kanade_step(pyramid_I1[level], I2_warp, window_size)
            u += du
            v += dv
            I2_warp = warp_image(pyarmid_I2[level], u, v)
        # Resize u and v to the next pyramid level resolution
        if level > 0:
            u = cv2.resize(u, (pyramid_I1[level - 1].shape)[::-1], interpolation=cv2.INTER_NEAREST) * 2
            v = cv2.resize(v, (pyramid_I1[level - 1].shape)[::-1], interpolation=cv2.INTER_NEAREST) * 2
    return u, v

def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object. The input video's VideoCapture.
    Returns:
        parameters: dict. A dictionary of parameters names to their values.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    parameters = {"fourcc": fourcc, "fps": fps, "height": height, "width": width, "frame_count": frame_count}
    return parameters


def get_gray_resized_frame(input_video_capture, width, height):  # TODO guy fix signature to be like others
    # Get the first frame from the input video
    retval, frame = input_video_capture.read()

    if not retval:
        print("Reading frame failed")
        return #TODO guy fix this to have a better return value

    # Convert the first frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (width, height))
    return frame

def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    # Open the input video capture
    input_video_capture = cv2.VideoCapture(input_video_path)

    # Open the input video capture and get its parameters
    parameters = get_video_parameters(input_video_capture)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = parameters["fps"]
    height = parameters["height"]
    width = parameters["width"]
    frame_count = parameters["frame_count"]

    # Open the output video writer
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), False)

    # Get the first frame from the input video
    retval, prev_frame = input_video_capture.read()

    # Convert it to grayscale
    gray_first_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Write the first frame to the output video
    output_video_writer.write(gray_first_frame)

    # Calculate the output video dims
    h_factor = int(np.ceil(gray_first_frame.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(gray_first_frame.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                    h_factor * (2 ** (num_levels - 1)))
    if gray_first_frame.shape != IMAGE_SIZE:
        gray_first_frame = cv2.resize(gray_first_frame, IMAGE_SIZE)


    # Create arrays to hold optical flow maps
    prev_u = np.zeros_like(gray_first_frame)
    prev_v = np.zeros_like(gray_first_frame)

    # Update the previous frame to be the first frame
    gray_prev_frame = gray_first_frame

    # Loop over frames in input video
    for i in tqdm(range(frame_count - 1)):
        # Read current frame
        retval, curr_gray_frame = input_video_capture.read()

        if not retval:
            print("Reading frame failed")
            break

        # Convert the current frame to grayscale
        curr_gray_frame = cv2.cvtColor(curr_gray_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_frame = cv2.resize(curr_gray_frame, IMAGE_SIZE)

        # Compute optical flow
        curr_u, curr_v = lucas_kanade_optical_flow(gray_prev_frame, curr_gray_frame, window_size, max_iter, num_levels)

        # Update u and v to their mean values inside the valid computation region
        w = window_size // 2
        curr_u[w:-w, w:-w] = np.mean(curr_u[w:-w, w:-w])
        curr_v[w:-w, w:-w] = np.mean(curr_v[w:-w, w:-w])

        # Add the u and v shift from the previous frame diff
        curr_u += prev_u
        curr_v += prev_v

        # Save current optical flow maps for next frame
        prev_u = curr_u.copy()
        prev_v = curr_v.copy()

        # Warp current frame using optical flow maps
        warped_frame = warp_image(curr_gray_frame, curr_u, curr_v)

        # Resize it back to the original frame size
        resized_warped_frame = cv2.resize(warped_frame, (width, height))

        # Write resized warped frame to output
        output_video_writer.write(resized_warped_frame)

        # Update the previous frame to be the current frame
        gray_prev_frame = curr_gray_frame

    # Release input and output videos
    input_video_capture.release()
    output_video_writer.release()
    cv2.destroyAllWindows()


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    # When the picture is small enough perform the good old lucas_kanade step
    if I1.shape[0] < 50 and I1.shape[1] < 50:
        return lucas_kanade_step(I1, I2, window_size)

    # Finding the corners in the picture I2
    corners = cv2.cornerHarris(I2.astype(np.float32), 5, 0, 0.05)
    corners = np.where(corners > 0.1 * corners.max(), 1, 0)
    corners = np.argwhere(corners == 1)

    kernel_t = np.array([[1], [1]])

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    # Compute half of window size
    w = window_size // 2

    for corner in corners:
        # for each point, calculate I_x, I_y, I_t
        Ix_Window = signal.convolve2d(I2[corner[0] - w:corner[0] + w + 1, corner[1] - w:corner[1] + w + 1],X_DERIVATIVE_FILTER, boundary='symm', mode='same')
        Iy_Window = signal.convolve2d(I2[corner[0] - w:corner[0] + w + 1, corner[1] - w:corner[1] + w + 1],Y_DERIVATIVE_FILTER, boundary='symm', mode='same')
        It_Window = I2.astype(int) - I1.astype(int)

        Ix_Vector = Ix_Window.reshape(-1)
        Iy_Vector = Iy_Window.reshape(-1)
        It_Vector = It_Window.reshape(-1)
        # Construct the A matrix and the b vector
        A = np.stack((Ix_Vector, Iy_Vector), axis=1)
        b = -It_Vector

        try:
            # Calculate the least squares solution, which is (u,v)
            least_squares_solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Extract du and dv from the least squares solution
            du[corner] = least_squares_solution[0]
            dv[corner] = least_squares_solution[1]
        except:
            # When the solution does not converge, keep this pixel's (u, v) as zero.
            pass

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)
    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)
    # Looping over every level in the image pyramid starting from the smallest image
    for level in range(num_levels, -1, -1):
        # Warp I2 from that level according to the current u and v
        I2_warp = warp_image(pyarmid_I2[level], u, v)
        # Repeat for num_iterations
        for iteration in range(max_iter):
            # Perform a Lucas Kanade Step with the I1 decimated image of the
            # current pyramid level and the current I2_warp to get the new I2_warp
            du, dv = faster_lucas_kanade_step(pyramid_I1[level], I2_warp, window_size)
            u += du
            v += dv
            I2_warp = warp_image(pyarmid_I2[level], u, v)
        # Resize u and v to the next pyramid level resolution
        if level > 0:
            u = cv2.resize(u, (pyramid_I1[level - 1].shape)[::-1], interpolation=cv2.INTER_NEAREST) * 2
            v = cv2.resize(v, (pyramid_I1[level - 1].shape)[::-1], interpolation=cv2.INTER_NEAREST) * 2
    return u, v


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    """INSERT YOUR CODE HERE."""
    # Open the input video capture
    input_video_capture = cv2.VideoCapture(input_video_path)

    # Open the input video capture and get its parameters
    parameters = get_video_parameters(input_video_capture)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = parameters["fps"]
    height = parameters["height"]
    width = parameters["width"]
    frame_count = parameters["frame_count"]

    # Open the output video writer
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), False)

    # Get the first frame from the input video
    retval, prev_frame = input_video_capture.read()

    # Convert it to grayscale
    gray_first_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Write the first frame to the output video
    output_video_writer.write(gray_first_frame)

    # Calculate the output video dims
    h_factor = int(np.ceil(gray_first_frame.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(gray_first_frame.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if gray_first_frame.shape != IMAGE_SIZE:
        gray_first_frame = cv2.resize(gray_first_frame, IMAGE_SIZE)

    # Create arrays to hold optical flow maps
    prev_u = np.zeros_like(gray_first_frame)
    prev_v = np.zeros_like(gray_first_frame)

    # Update the previous frame to be the first frame
    gray_prev_frame = gray_first_frame

    # Loop over frames in input video
    for i in tqdm(range(frame_count - 1)):
        # Read current frame
        retval, curr_gray_frame = input_video_capture.read()

        if not retval:
            print("Reading frame failed")
            break

        # Convert the current frame to grayscale
        curr_gray_frame = cv2.cvtColor(curr_gray_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_frame = cv2.resize(curr_gray_frame, IMAGE_SIZE)

        # Compute optical flow
        curr_u, curr_v = faster_lucas_kanade_optical_flow(gray_prev_frame, curr_gray_frame, window_size, max_iter, num_levels)

        # Update u and v to their mean values inside the valid computation region
        w = window_size // 2
        curr_u[w:-w, w:-w] = np.mean(curr_u[w:-w, w:-w])
        curr_v[w:-w, w:-w] = np.mean(curr_v[w:-w, w:-w])

        # Add the u and v shift from the previous frame diff
        curr_u += prev_u
        curr_v += prev_v

        # Save current optical flow maps for next frame
        prev_u = curr_u.copy()
        prev_v = curr_v.copy()

        # Warp current frame using optical flow maps
        warped_frame = warp_image(curr_gray_frame, curr_u, curr_v)

        # Resize it back to the original frame size
        resized_warped_frame = cv2.resize(warped_frame, (width, height))

        # Write resized warped frame to output
        output_video_writer.write(resized_warped_frame)

        # Update the previous frame to be the current frame
        gray_prev_frame = curr_gray_frame

    # Release input and output videos
    input_video_capture.release()
    output_video_writer.release()
    cv2.destroyAllWindows()


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    # Open the input video capture
    input_video_capture = cv2.VideoCapture(input_video_path)

    # Open the input video capture and get its parameters
    parameters = get_video_parameters(input_video_capture)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = parameters["fps"]
    height = parameters["height"]
    width = parameters["width"]
    frame_count = parameters["frame_count"]

    # Open the output video writer
    output_video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), False)

    # Get the first frame from the input video
    retval, prev_frame = input_video_capture.read()

    # Convert it to grayscale
    gray_first_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the output video dims
    h_factor = int(np.ceil(gray_first_frame.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(gray_first_frame.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if gray_first_frame.shape != IMAGE_SIZE:
        gray_first_frame = cv2.resize(gray_first_frame, IMAGE_SIZE)

    # Create arrays to hold optical flow maps
    prev_u = np.zeros_like(gray_first_frame)
    prev_v = np.zeros_like(gray_first_frame)

    # Update the previous frame to be the first frame
    gray_prev_frame = gray_first_frame

    # Loop over frames in input video
    for i in tqdm(range(frame_count - 1)):
        # Read current frame
        retval, curr_gray_frame = input_video_capture.read()

        if not retval:
            print("Reading frame failed")
            break

        # Convert the current frame to grayscale
        curr_gray_frame = cv2.cvtColor(curr_gray_frame, cv2.COLOR_BGR2GRAY)
        curr_gray_frame = cv2.resize(curr_gray_frame, IMAGE_SIZE)

        # Compute optical flow
        curr_u, curr_v = faster_lucas_kanade_optical_flow(gray_prev_frame, curr_gray_frame, window_size, max_iter,
                                                          num_levels)

        # Update u and v to their mean values inside the valid computation region
        w = window_size // 2
        curr_u[w:-w, w:-w] = np.mean(curr_u[w:-w, w:-w])
        curr_v[w:-w, w:-w] = np.mean(curr_v[w:-w, w:-w])

        # Add the u and v shift from the previous frame diff
        curr_u += prev_u
        curr_v += prev_v

        # Save current optical flow maps for next frame
        prev_u = curr_u.copy()
        prev_v = curr_v.copy()

        # Warp current frame using optical flow maps and cut it
        warped_frame = warp_image(curr_gray_frame, curr_u, curr_v)
        gray_prev_frame = curr_gray_frame
        curr_cut_frame = frame_cutter(warped_frame, start_rows, end_rows, start_cols, end_cols)

        # Resize it back to the original frame size
        resized_warped_cut_frame = cv2.resize(curr_cut_frame, (width, height))

        # Write resized cut warped frame to output
        output_video_writer.write(resized_warped_cut_frame)

        # Update the previous frame to be the current frame
        # gray_prev_frame = curr_gray_frame
        # curr_cut_frame = cut_frame(resized_warped_frame, start_rows, end_rows, start_cols, end_cols)
    # Release input and output videos
    input_video_capture.release()
    output_video_writer.release()
    cv2.destroyAllWindows()


# supporting function to perform the borders cut
def frame_cutter(input_frame, start_rows, end_rows, start_cols, end_cols):
    output_frame = input_frame[start_rows:-end_rows, start_cols:-end_cols]
    return output_frame

