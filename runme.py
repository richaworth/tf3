# from tf3_yolo_bounds.create_yolo_bounds_init import main as create_bounds
from tf3_yolo_bounds.create_yolo_bounds_teeth import main as create_teeth
from tf3_yolo_bounds.create_yolo_bounds_teeth_sliced import main as create_teeth_2

# from tf3_yolo_bounds.train_yolo_bounds_finder import main as train_bounds
from tf3_yolo_bounds.train_yolo_teeth_finder import main as train_teeth
from tf3_yolo_bounds.train_yolo_teeth_finder import main as train_teeth_2

from tf3_train_per_tooth import main as train_tooth_3d
from tf3_train_bridges_and_crowns import main as train_bridge_3d
from tf3_train_jawbones import main as train_jaw_3d
from tf3_train_sinuses_pharynx import main as train_sinus_3d

def main():
    # create_bounds train_bounds, create_teeth, create_teeth_2, train_teeth, train_teeth_2, 

    list_run = [train_tooth_3d, train_bridge_3d, train_jaw_3d, train_sinus_3d]

    for func in list_run:
        try:    
            func()
        except:
            continue
        


if __name__ == "__main__":
    main()


