# Phragmites_Classification

### Author: Matthew Walter
### Email: mswalter@udel.edu

This is a GEE code used to classify the invasive species Phragmites using National Agriculture Imagery Program (NAIP) data.
Three different inputs were tested for accuracy:
1. Four NAIP bands (N,R,G,B)
2. Four NAIP bands and NDVI
3. 5 PCA bands derived from NAIP and NDVI

Input 3 yielded the highest accuracy followed by 2 and 1.

Running the code involves the input of three different variables:
1. region: Study area geometry
2. table: Feature collection containing reference points
3. label: The field which contains the landcover IDs from the reference point feature collection
