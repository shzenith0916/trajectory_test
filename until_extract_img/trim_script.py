# -*- coding: utf-8 -*-

import os
import ffmpeg

# ffmpeg -i input.mp4 -ss 00:04:40 -to 00:04:50 output.mp4


def trim_video(input_file, output_file, start, end):
    (
        ffmpeg
        .input(input_file, ss=start, to=end)
        # 'codec="copy"' applies to both video and audio. Work as the line -vcodec -acodec in FFmpeg.
        # Without the codec line, the video quality is not assured.
        .output(output_file, codec="copy")
        .run(capture_stdout=True, capture_stderr=True)
    )


if __name__ == '__main__':
    # start time and end time setting,  HH:MM:SS format
    print("Please type the start second")
    startsecond = input()
    st = "00:00:{}".format(startsecond)
    print("Please type the end second")
    endsecond = input()
    et = "00:00:{}".format(endsecond)

    print("Please insert input file name")
    input_file = input()

    file_name, file_extension = os.path.splitext(input_file)
    output_file = file_name + "_trimmed" + file_extension

    # Call the method
    trim_video(input_file, output_file, st, et)