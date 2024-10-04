from itertools import groupby
from sys import argv, exit

from PIL import Image
from PIL import ImageDraw

# Parse Command Line Arguments
if len(argv) < 3:
    exit("Usage is render.py <input file> <output file> [cutoff]")

input_file = argv[1]
output_file = argv[2]
# cutoff = None if len(argv) <= 3 else float(argv[3])


# Helper Functions
def circle_to_box(center_x, center_y, size):
    return (center_x - size, center_y - size,
            center_x + size, center_y + size)


# Process the file
with open(input_file, 'r') as f:
    # Get first line to find the number of particles and the box size.
    num_parts, d_IU, x_min, x_max, y_min, y_max = next(f).split()
    num_parts, d_IU, x_min, x_max, y_min, y_max = int(num_parts), float(d_IU), float(x_min), float(x_max), float(y_min), float(y_max)
    box_size = max(x_max-x_min, y_max-y_min)
    box_size_x = x_max-x_min
    box_size_y = y_max-y_min

    max_pixel_length = 4000

    if box_size_x == box_size_y:
        pixels_x = max_pixel_length
        pixels_y = max_pixel_length
    elif box_size_x < box_size_y:
        pixels_y = max_pixel_length
        pixels_x = int(max_pixel_length*box_size_x/box_size_y)
    elif box_size_y < box_size_x:
        pixels_x = max_pixel_length
        pixels_y = int(max_pixel_length*box_size_y/box_size_x)


    # # Compute cutoff_radius
    # cutuff_radius = int(1024 * ((cutoff or 0) / cutoff))

    # Parse input file
    frames = []
    file_sections = groupby(f, lambda x: x and not x.isspace())

    frame_sections = (x[1] for x in file_sections if x[0])

    margin = 200
    time_step = 0
    for frame_section in frame_sections:
        if time_step%5 == 0 and time_step < 800:
        # if time_step == 540:
            # Set up a new frame
            img = Image.new("RGBA", (pixels_x+margin, pixels_y+margin), 'white')
            drawer = ImageDraw.Draw(img)
            frames.append(img)

            # Paint in the frame
            lines = 0
            for line in frame_section:
                center_x, center_y, status = line.split()
                center_x = int(max_pixel_length * ((float(center_x)-x_min) / box_size)) + margin/2
                center_y = int(max_pixel_length * ((float(center_y)-y_min) / box_size))+ margin/2
                status = int(status)

                if status == 0:
                    drawer.ellipse(circle_to_box(center_x, center_y, int(d_IU*max_pixel_length)), 'deepskyblue')
                elif status == 1:
                    drawer.ellipse(circle_to_box(center_x, center_y, int(d_IU*max_pixel_length)), 'darkorange')
                elif status == 2:
                    drawer.ellipse(circle_to_box(center_x, center_y, int(d_IU*max_pixel_length)), 'red')
                elif status == 3:
                    drawer.ellipse(circle_to_box(center_x, center_y, int(d_IU*max_pixel_length)), 'lightgrey')

                # drawer.ellipse(circle_to_box(center_x, center_y, 1), 'black')

            #only appropriate for specific figure generation: hardcoded
            sp1_xmin = 0 + margin/2
            sp1_xmax = (0.05/x_max)*max_pixel_length + margin/2
            sp1_ymin = 0 + margin/2
            sp1_ymax = pixels_y + margin/2
            sp2_xmin = (0.053/x_max)*max_pixel_length +margin/2
            sp2_xmax = (0.103/x_max)*max_pixel_length +margin/2
            sp2_ymin = 0 + margin/2
            sp2_ymax = pixels_y + margin/2


            drawer.rectangle([sp1_xmax, sp1_ymin, sp2_xmin, sp1_ymax], fill ="white", outline ="white") #adding white rectange between subpops to cut off circles outside of boundary
            drawer.rectangle([0, 0, sp1_xmin, pixels_y+margin], fill ="white", outline ="white") #adding white rectange between subpops to cut off circles outside of boundary
            drawer.rectangle([0, 0, pixels_x+margin, sp1_ymin], fill ="white", outline ="white") #adding white rectange between subpops to cut off circles outside of boundary
            drawer.rectangle([sp2_xmax, 0, pixels_x+margin, pixels_y+margin], fill ="white", outline ="white") #adding white rectange between subpops to cut off circles outside of boundary
            drawer.rectangle([0, sp1_ymax, pixels_x+margin, pixels_y+margin], fill ="white", outline ="white") #adding white rectange between subpops to cut off circles outside of boundary
            drawer.rectangle([sp1_xmin, sp1_ymin, sp1_xmax, sp1_ymax], fill = None, outline ="black", width = 10) 
            drawer.rectangle([sp2_xmin, sp2_ymin, sp2_xmax, sp2_ymax], fill = None, outline ="black", width = 10) 
            # drawer.line([sp1_xmin, sp1_ymin, sp, pixels_y], fill='red', width=10) #draw borders around subpops
            # drawer.line([(0.1/x_max)*max_pixel_length, 0, (0.1/x_max)*max_pixel_length, pixels_y], fill='red', width=10) #draw borders around subpops
            # drawer.line([0, pixels_y, (0.1/x_max)*max_pixel_length, pixels_y], fill='red', width=10) #draw borders around subpops
            # drawer.line([0, 0, (0.1/x_max)*max_pixel_length, 0], fill='red', width=10) #draw borders around subpops
            # drawer.line([(0.14/x_max)*max_pixel_length, 0, (0.14/x_max)*max_pixel_length, pixels_y], fill='red', width=10) #draw borders around subpops
            # drawer.line([(0.24/x_max)*max_pixel_length, 0, (0.24/x_max)*max_pixel_length, pixels_y], fill='red', width=10) #draw borders around subpops
            # drawer.line([(0.14/x_max)*max_pixel_length, pixels_y, (0.24/x_max)*max_pixel_length, pixels_y], fill='red', width=10) #draw borders around subpops
            # drawer.line([(0.14/x_max)*max_pixel_length, 0, (0.24/x_max)*max_pixel_length, 0], fill='red', width=10) #draw borders around subpops
            
        time_step +=1
        print(time_step)
    frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

    # frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)
    # frames[0].save(output_file, format='png', append_images=frames[1:], save_all=True, duration=100, loop=0)
