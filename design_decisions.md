- use nii.gz (compressed nifty, same as FSL uses)
- use simple itk for loading/reading mha images, permuate the axes from x,y,z to z,y,x to match niibabel
- use nibabel for loading/reading nii images
  - When saving/loading there is a small float difference due to float rounding precision
- use pathlib to define all paths

not yet implemented
use for images 0 - 255 range (uint8) and for masks 0 - c range (uint8)


mha: int16
nii: int16







To run testing with a reporting of coverage use:
pip install pytest-cov

```
pytest --cov-report term-missing --cov=src Tests/
```



things to consider:
- using @typechecked decorator
- checking validity of arguments at start of function\
- whether to accept strings as input to read/write functions
- how to test image similarity with different transforms