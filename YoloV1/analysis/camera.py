from analysis.predictor import Predictor
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":

    predictor = Predictor('/home/alex/workspace/experiments/best_model/model_metric.pth.tar')


    # define a video capture object
    vid = cv2.VideoCapture(0)
    fig, ax = plt.subplots(1)

    freq = 5
    need = freq
    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()


        # Display the resulting frame
        if need == freq:
            need = 0
            frame = predictor.opencv_img(frame)


            # Display the image
            ax.imshow(frame)
            plt.pause(0.001)

        need = need + 1

       # frame  = cv2.resize(frame, (height, width), interpolation = cv2.INTER_AREA)
        #cv2.imshow('frame', frame2)


        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()