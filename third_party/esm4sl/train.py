from coach_pl.tool.train import arg_parser, main

# Import this module to register the datasets, models, and modules.
import esm4sl

# Use the default training script.
if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(args)
