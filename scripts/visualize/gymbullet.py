output = None
if record_video:
    # Display simulation result
    import glob
    import os
    from PIL import Image as pilImage

    def make_gif(frame_folder):
        frames = []
        try:
            i = 0
            while i > -1:
                frames += [
                    pilImage.open(os.path.join(f"{frame_folder}", f"frame_{i}.png"))
                ]
                i += 1
        except Exception as e:
            pass

        frame_one = frames[0]
        frame_one.save(
            "example_output.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=3000 // len(frames),
            loop=0,
        )

    videos = glob.glob("/content/MARL-Drones/scripts/learning/results/recording_*/")
    videos.sort()
    make_gif(videos[-1])

    from IPython.display import Image as ipyImage

    output = ipyImage(open("example_output.gif", "rb").read())
output
