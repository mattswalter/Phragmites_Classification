/// Author: Matthew Walter, mswalter@udel.edu
/// Code to classify the invasive species Phragmites using NAIP data
/// Input parameters needed:
  // table = table which has the classification points for your different classes
    // label = the field name where the land cover IDs are stored
  // region = a geometry for the study area
var table = phrag1.merge(marsh_other).merge(water)
var label = 'class'; 

/// Load in and filter NAIP imagery for 2017

var data_no_ndvi = ee.ImageCollection('USDA/NAIP/DOQQ')
                  .filterBounds(region)
                  .filter(ee.Filter.date('2017-01-01', '2017-10-31'))
                  .min()
                  .clip(region);
                  
/// Add NAIP true color image to the map
var trueColor = data_no_ndvi.select(['R', 'G', 'B']);
var trueColorVis = {
  min: 0.0,
  max: 255.0,
};
Map.addLayer(data_no_ndvi,trueColorVis,'NAIP True')

/// Calculate NDVIusing the near infrared and red bands
var ndvi = data_no_ndvi.normalizedDifference(['N', 'R']).rename('NDVI');

/// Map NDVI
var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
Map.addLayer(ndvi, ndviParams, 'NDVI NAIP');


/// Calculate PCA bands
var data_ndvi = ndvi.addBands(data_no_ndvi)
// Display the input imagery and the region in which to do the PCA.
var region = data_ndvi.geometry();

Map.addLayer(ee.Image().paint(region, 0, 2), {}, 'Region');


// Set some information about the input to be used later.
var scale = 30;
var bandNames = data_ndvi.bandNames();

// Mean center the data to enable a faster covariance reducer
// and an SD stretch of the principal components.
var meanDict = data_ndvi.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: scale,
    maxPixels: 1e9
});
var means = ee.Image.constant(meanDict.values(bandNames));
var centered = data_ndvi.subtract(means);

// This helper function returns a list of new band names.
var getNewBandNames = function(prefix) {
  var seq = ee.List.sequence(1, bandNames.length());
  return seq.map(function(b) {
    return ee.String(prefix).cat(ee.Number(b).int());
  });
};

// This function accepts mean centered imagery, a scale and
// a region in which to perform the analysis.  It returns the
// Principal Components (PC) in the region as a new image.
var getPrincipalComponents = function(centered, scale, region) {
  // Collapse the bands of the image into a 1D array per pixel.
  var arrays = centered.toArray();

  // Compute the covariance of the bands within the region.
  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: region,
    scale: scale,
    maxPixels: 1e9
  });

  // Get the 'array' covariance result and cast to an array.
  // This represents the band-to-band covariance within the region.
  var covarArray = ee.Array(covar.get('array'));

  // Perform an eigen analysis and slice apart the values and vectors.
  var eigens = covarArray.eigen();

  // This is a P-length vector of Eigenvalues.
  var eigenValues = eigens.slice(1, 0, 1);
  // This is a PxP matrix with eigenvectors in rows.
  var eigenVectors = eigens.slice(1, 1);

  // Convert the array image to 2D arrays for matrix computations.
  var arrayImage = arrays.toArray(1);

  // Left multiply the image array by the matrix of eigenvectors.
  var principalComponents = ee.Image(eigenVectors).matrixMultiply(arrayImage);

  // Turn the square roots of the Eigenvalues into a P-band image.
  var sdImage = ee.Image(eigenValues.sqrt())
    .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);

  // Turn the PCs into a P-band image, normalized by SD.
  return principalComponents
    // Throw out an an unneeded dimension, [[]] -> [].
    .arrayProject([0])
    // Make the one band array image a multi-band image, [] -> image.
    .arrayFlatten([getNewBandNames('pc')])
    // Normalize the PCs by their SDs.
    .divide(sdImage);
};

// Get the PCs at the specified scale and in the specified region
var pcImage = getPrincipalComponents(centered, scale, region);

for (var i = 0; i < bandNames.length().getInfo(); i++) {
  var band = pcImage.bandNames().get(i).getInfo();}
print(pcImage)

/// Add PCA bands to the map
var pc1 = pcImage.select(['pc1'])
var pc1_clip = pc1.clip(region)
Map.addLayer(pc1_clip, {min: -2, max: 2}, 'PC1')

var pc2 = pcImage.select(['pc2'])
var pc2_clip = pc2.clip(region)
Map.addLayer(pc2_clip, {min: -2, max: 2}, 'PC2')

var pc3 = pcImage.select(['pc3'])
var pc3_clip = pc3.clip(region)
Map.addLayer(pc3_clip, {min: -2, max: 2}, 'PC3')

var pc4 = pcImage.select(['pc4'])
var pc4_clip = pc4.clip(region)
Map.addLayer(pc4_clip, {min: -2, max: 2}, 'PC4')

var pc5 = pcImage.select(['pc5'])
var pc5_clip = pc5.clip(region)
Map.addLayer(pc5_clip, {min: -2, max: 2}, 'PC5')

// Merge all PCA bands
var pca = (pc1_clip).addBands(pc2_clip).addBands(pc3_clip).addBands(pc4_clip).addBands(pc5_clip)


//////// Select which variable to incude in the model from the following 3:
/// RF Model 1 variable: just 4 NAIP bands
// data_no_ndvi

/// RF Model 2 variable: 4 NAIP bands with NDVI
// data_ndvi

/// RF Model 3 variable: 5 PCA bands
// pca

var dataset_m = pca

/// Randomly subset each of the landcover classes into 70 percent for testing the model and 30 percent for training/validation
// Class 1 - Phragmites
var c1 = (table.filter(ee.Filter.eq('class', 1)));
var c1 = c1.randomColumn();
var test_c1 = c1.filter(ee.Filter.lt('random', 0.7));
var val_c1 = c1.filter(ee.Filter.gte('random', 0.7));
// Class 2 - Other marsh vegetation
var c2 = (table.filter(ee.Filter.eq('class', 2)));
var c2 = c2.randomColumn();
var test_c2 = c2.filter(ee.Filter.lt('random', 0.7));
var val_c2 = c2.filter(ee.Filter.gte('random', 0.7));
// Class 3 - Open water
var c3 = (table.filter(ee.Filter.eq('class', 3)));
var c3 = c3.randomColumn();
var test_c3 = c3.filter(ee.Filter.lt('random', 0.7));
var val_c3 = c3.filter(ee.Filter.gte('random', 0.7));

/// Merge the testing points into one variable
var all_merged_points_test = test_c1.merge(test_c2).merge(test_c3);

/// Train RF classifier
var training = dataset_m.sampleRegions({
collection: all_merged_points_test,
properties: [label],
scale: 1 
});
var trained = ee.Classifier.smileRandomForest(100).train(training,label);
var classified = dataset_m.classify(trained);

/// Map classified image
Map.addLayer(classified,
{min: 1, max: 3, palette: ['#171A1C', '#4E873D', '#AFBFDC']},'classification');

/// Merge the validation points into one variable
var all_merged_points_val = val_c1.merge(val_c2).merge(val_c3);

/// Calculate eror matrix from training data
var trainAccuracy = trained.confusionMatrix();
print('Resubstitution error matrix: ', trainAccuracy);
print('Training overall accuracy: ', trainAccuracy.accuracy());

/// Sample the input with a different random seed to get validation data.
var validation = dataset_m.sampleRegions({
    collection: all_merged_points_val,
    properties: [label],
    scale: 1  //scale is pixel
});

/// Classify the validation data.
var validated = validation.classify(trained);

/// Get a confusion matrix representing expected accuracy.
var testAccuracy = validated.errorMatrix('class', 'classification');
print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ', testAccuracy.accuracy());

var classifier = trained

/// Calculate the variable importance
var dataset_m = dataset_m.float()
var dict = classifier.explain();
print('Explain:',dict);

var variable_importance = ee.Feature(null, ee.Dictionary(dict).get('importance'));

var chart =
  ui.Chart.feature.byProperty(variable_importance)
    .setChartType('ColumnChart')
    .setOptions({
      title: 'Random Forest Variable Importance',
      legend: {position: 'none'},
      hAxis: {title: 'Bands'},
      vAxis: {title: 'Importance'}
    });


print(chart); 


// Export the classified image
Export.image.toDrive({
  image: classified.clip(estuarine_wetlands),
  description: 'phragmites_classified_11_2_22',
  maxPixels: 91180135690 ,
  scale: 1,
  region: region
});

// Export the reference point table
Export.table.toDrive({
  collection: all_merged_points_val,
  description:'phragmites_test_points_11_2_22',
  fileFormat: 'SHP'
});
