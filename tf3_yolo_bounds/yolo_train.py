from create_yolo_bounds_init import main as create_bounds
from create_yolo_bounds_teeth import main as create_teeth
from create_yolo_bounds_teeth_sliced import main as create_teeth_2

from train_yolo_bounds_finder import main as train_bounds
from train_yolo_teeth_finder import main as train_teeth
from train_yolo_teeth_finder import main as train_teeth_2

def main():
    # create_bounds()
    # train_bounds()

    create_teeth()
    create_teeth_2()

    train_teeth()
    train_teeth_2()

if __name__ == "__main__":
    main()


