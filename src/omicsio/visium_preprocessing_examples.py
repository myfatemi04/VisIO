from .datasets import load_slide_from_folder, load_old_visium_from_folder
from .large_images import load_image, crop_large_image_for_slide, downsample_image

def cache_autostainer_slides():
    pre = 'input_data/autostainer/final_data/Autostainer_20x'
    slide_40x = load_slide_from_folder(pre, 'input_data/autostainer/autostain_images/_SS12254_081342_40x.tiff', spot_image_scaling=2)
    slide_40x.save('input_data/preprocessed/autostainer_40x.pkl')

def cache_manual_slides():
    pre = 'input_data/autostainer/final_data/Manual'
    slide_20x = load_slide_from_folder(pre, 'input_data/autostainer/visium_data_half_sequenced/images/EVOS/Manual_Epredia.TIF')
    slide_20x.save('input_data/preprocessed/manual_20x.pkl')    

def cache_visium_slides():
    for slide_id in ['A1', 'B1', 'C1', 'D1']:
        print("Caching Visium slide", slide_id)

        slide = load_old_visium_from_folder(f'input_data/visium/raw_data/{slide_id}', f'input_data/visium/raw_data/{slide_id}.TIF')
        slide.save(f'input_data/preprocessed/visium_{slide_id}.pkl')

def cache_new_colon_visium_full_sequenced():
    slide_ids = ['091759', '092146', '092534', '092842']
    # Alignment of image names to slide ids is based on the *.sbatch files in the new_colon_visium_full_sequenced/analysis/ folder.
    image_names = ['A_Sample3', 'B_Sample4', 'A_Sample1', 'B_Sample2']

    for slide_id, image_name in zip(slide_ids[:1], image_names[:1]):
        print("Running on slide", slide_id)

        pre = 'input_data/new_colon_visium_full_sequenced/analysis/' + slide_id

        slide_40x = load_slide_from_folder(pre, 'input_data/new_colon_visium_full_sequenced/images/Scanner/' + image_name + '.svs', spot_image_scaling=2)
        slide_40x_image = load_image(slide_40x.image_path)
        slide_40x_image, slide_40x.spot_locations = crop_large_image_for_slide(slide_40x_image, slide_40x.spot_locations, buffer=2048)
        slide_40x_image.save('input_data/preprocessed/new_colon_visium_40x_' + slide_id + '_crop.tiff')
        slide_40x.save('input_data/preprocessed/new_colon_visium_40x_' + slide_id + '.pkl')

        slide_20x = load_slide_from_folder(pre, 'input_data/new_colon_visium_full_sequenced/images/Scanner/' + image_name + '.svs', spot_image_scaling=1)
        slide_20x_image, slide_20x.spot_locations = downsample_image(slide_40x_image, slide_40x.spot_locations)
        slide_20x_image.save('input_data/preprocessed/new_colon_visium_20x_' + slide_id + '_crop.tiff')
        slide_20x.save('input_data/preprocessed/new_colon_visium_20x_' + slide_id + '.pkl')

    print("Done caching new_colon_visium_full_sequenced")

def cache_slides():
    cache_visium_slides()
    cache_autostainer_slides()
    cache_manual_slides()
    cache_new_colon_visium_full_sequenced()
