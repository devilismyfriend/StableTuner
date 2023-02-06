import sys
import os
try:
    import converters
except ImportError:

    #if there's a scripts folder where the script is, add it to the path
    if 'scripts' in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\\scripts')
    else:
        print('Could not find scripts folder. Please add it to the path manually or place this file in it.')
    import converters


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 4:
        print('Usage: python3 convert_sd_to_diffusers.py <model_path> <output_path> <version> <prediction_type>')
        sys.exit(1)
    checkpoint_path = args[0]
    output_path = args[1]
    version = args[2]
    prediction_type = args[3]
    converters.Convert_SD_to_Diffusers(
        checkpoint_path,
        output_path,
        version = version,
        prediction_type = prediction_type
    )