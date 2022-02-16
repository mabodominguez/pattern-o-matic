# pattern-o-matic
This is an app to allow projection of sewing PDF patterns onto fabric, created for CSCI154, Computer Vision at Harvey Mudd College. I used EasyOCR and tkinter to do OCR on the pattern pieces and create a graphical user interface.

My method to take a sewing pattern and information on fabric yardage and process it so it can be projected onto the length of fabric for easy cutting can be broken into three parts.
  * Segmentation of pattern pieces from an image of a sewing pattern.
  * Parsing information on grainline to orient the pattern shapes
  * Using orientation information from pattern, allow users pack pattern pieces into a minimal packing into the fabric yardage.
  
Feel free to look at my final report of the project [here](https://github.com/mabodominguez/pattern-o-matic/blob/main/Computer_Vision_Project_Final_Report.pdf), and take a look at the final demo below.


https://user-images.githubusercontent.com/60116262/154178402-b54ba587-9daa-42dc-85fa-ffe87501d0df.mov

