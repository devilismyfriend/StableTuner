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
    if len(args) != 2:
        print('Usage: python3 convert_diffusers_to_sd.py <model_path> <output_path>')
        sys.exit(1)
    model_path = args[0]
    output_path = args[1]
    converters.Convert_Diffusers_to_SD(model_path, output_path)
