from PIL import Image

# Open the image
img = Image.open('/Users/shanxiafeng/Desktop/brain.png')

# Convert the image to RGBA (if not already in that mode)
img = img.convert("RGBA")

# Get the data of the image
data = img.getdata()

# Create a new list for the new data
new_data = []

# Define the background color you want to make transparent
background_color = (255, 255, 255, 255)  # White background

# Loop through the data and change the background color to transparent
for item in data:
    if item[:3] == background_color[:3]:  # Check RGB value (ignore alpha)
        new_data.append((255, 255, 255, 0))  # Change to transparent
    else:
        new_data.append(item)  # Keep original color

# Update image data
img.putdata(new_data)

# Save the image with a new name
img.save('FigureTable/manuscript_figures/fig1_brain_image_transparent.png', 'PNG')