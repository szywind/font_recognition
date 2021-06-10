nohup python -u trdg/run_font.py \
      --output_dir /home/t-wmiao/PyProj/data/font_classify_data/0528_gen/ \
      --label_dir ./labels/ \
      --count 10000 \
      --thread_count 4 \
      --blur 1 \
      --random_blur \
      --font_dir ./trdg/fonts_for_classify/cn \
      --image_dir ./bg_imgs/ > gen_fonts_classify_log.txt &