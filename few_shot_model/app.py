import streamlit as st
import os
import subprocess
import cv2
from skimage import metrics

# Create temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Streamlit UI
st.title("Real-ESRGAN Image Enhancer")

# File upload for image
uploaded_file = st.file_uploader("Choose an image to enhance", type=["png", "jpg", "jpeg"])

# File upload for .pth model file
uploaded_model = st.file_uploader("Upload a .pth model file ", type=["pth"])

if uploaded_file:
    # Define paths for input and output images
    input_path = os.path.join("temp", uploaded_file.name)
    output_path = os.path.join("temp", "output.png")

    # Save the uploaded file to the specified path
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(input_path, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Enhance Image"):
        # Ensure Real-ESRGAN executable path is correct
        command = f'realesrgan-ncnn-vulkan.exe -i "{input_path}" -o "{output_path}"'
        subprocess.run(command, shell=True)

        # Display the enhanced image
        if os.path.exists(output_path):
            st.image(output_path, caption="Enhanced Image", use_column_width=True)
            st.success("Image enhanced successfully!")

            # Calculate and display PSNR and SSIM
            enhanced_image = cv2.imread(output_path)
            ground_truth_image = cv2.imread(input_path)  # Update this with actual ground truth if available

            if enhanced_image is not None and ground_truth_image is not None:
                # Resize enhanced image to match ground truth dimensions
                enhanced_image_resized = cv2.resize(enhanced_image, (ground_truth_image.shape[1], ground_truth_image.shape[0]))

                # Convert images to grayscale
                enhanced_image_gray = cv2.cvtColor(enhanced_image_resized, cv2.COLOR_BGR2GRAY)
                ground_truth_image_gray = cv2.cvtColor(ground_truth_image, cv2.COLOR_BGR2GRAY)

                # Calculate PSNR
                psnr_value = metrics.peak_signal_noise_ratio(ground_truth_image_gray, enhanced_image_gray)

                # Calculate SSIM
                ssim_value = metrics.structural_similarity(ground_truth_image_gray, enhanced_image_gray)

                # Print the PSNR and SSIM values
                st.write(f'PSNR: {psnr_value:.2f} dB')
                st.write(f'SSIM: {ssim_value:.4f}')
            else:
                st.error("Error loading images for PSNR and SSIM calculation.")
        else:
            st.error("There was an issue enhancing the image.")
