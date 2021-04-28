import matplotlib.pyplot as plt


def display_rgbd(images):
    # Displays a rgb image and a depth image side by side
    # Can take more than two images
    plt.figure(figsize=(15, len(images) * 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        if i < 1:
            plt.title("RGB image")
        else:
            plt.title("Depth image")
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()
    return None
