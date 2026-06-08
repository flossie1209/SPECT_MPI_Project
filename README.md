# SPECT_MPI_Project

Coronary artery disease, or CAD, is one of the leading causes of cardiovascular morbidity and mortality worldwide. It occurs when the coronary arteries become narrowed or blocked, reducing blood flow and oxygen delivery to the heart muscle. If left untreated, this can lead to myocardial ischemia, chest pain, heart failure, or myocardial infarction.
To evaluate myocardial perfusion and detect ischemia, clinicians commonly use SPECT myocardial perfusion imaging, or SPECT MPI. This non-invasive imaging modality provides functional information about blood flow within the myocardium by comparing stress and rest perfusion patterns.
However, interpreting SPECT MPI scans can be challenging. Conventional analysis is often time-consuming and subject to inter-reader variability. In addition, imaging artifacts and subtle perfusion abnormalities can make diagnosis difficult, potentially leading to false positives or false negatives.
These challenges create an opportunity for artificial intelligence. Deep learning models can automatically learn complex spatial patterns from stress-rest SPECT MPI images and assist clinicians in distinguishing normal from abnormal perfusion. By improving diagnostic consistency and efficiency, AI has the potential to support earlier and more accurate detection of CAD-related ischemia.
This clinical need motivated our project, where we developed and evaluated deep learning models for automated classification of SPECT MPI images.

Our first objective was to develop an automated deep learning framework capable of analyzing SPECT myocardial perfusion images and classifying them as normal or abnormal.
The second objective was to investigate whether transfer learning with ResNet18 could outperform a custom CNN architecture for this classification task. By comparing both approaches, we aimed to determine whether leveraging pretrained image features provides a performance advantage for SPECT MPI data.
Our third objective was to improve model interpretability using Grad-CAM. Since deep learning models are often considered black boxes, Grad-CAM helps visualize which image regions contribute most to the model's prediction, increasing clinical transparency and trust

For this project, we used a public SPECT MPI dataset with stress-rest technetium-99m myocardial perfusion imaging scans from 192 patients. The dataset is imbalanced, with 150 CAD patients and 42 non-CAD patients. Scans with ischemia and/or infarction were labeled as abnormal, while scans without perfusion defects were labeled as normal. Looking at the two examples in the middle. Clinically, ischemia is usually suggested when a perfusion defect appears during stress but is reduced or absent at rest. In contrast, normal scans show more uniform radiotracer distribution across both stress and rest images.
For input, we used the full SPECT MPI image as one image-level input instead of separating stress and rest into two channels. This allowed the CNN to learn spatial and intensity patterns across the full scan layout.
For preprocessing, images were resized to 224 by 224 pixels, converted to RGB tensor format, and normalized using ImageNet values to match the ResNet18 input format. During training, we applied random rotation within plus or minus 10 degrees and horizontal flipping for augmentation. Because the dataset was imbalanced, we also used class weighting and weighted sampling to reduce bias toward the majority class.

Next, for the CNN architecture, we compared two different approaches. The first was a custom CNN trained from scratch, which we used as a baseline model. This helped us test whether a basic CNN could learn useful features from the SPECT MPI images.

The second approach was a ResNet18 transfer learning model, which we selected as our final model. We chose ResNet18 because it is a stronger architecture for image feature extraction, and transfer learning can be helpful when the dataset is relatively small.

For the final pipeline, the ResNet18 model was initialized with ImageNet-pretrained weights. Even though ImageNet is not a medical imaging dataset, the early convolutional layers can still learn general visual features, such as edges, textures, shapes, and intensity patterns.

We then replaced the final fully connected layer so the model could output two classes: normal and abnormal. The final prediction was based on which class had the higher predicted probability.

For training, we used class-weighted cross-entropy loss to address the imbalance between normal and abnormal images. We used the Adam optimizer with a learning rate of 1 times 10 to the -4, a batch size of 16, and trained the model for 20 epochs. During training, we monitored validation accuracy and saved the model checkpoint with the best validation performance.

Grad-CAM performs a forward pass through ResNet18 to obtain a prediction. The gradients of the predicted class are then backpropagated to the final convolutional layer. These gradients are globally averaged to compute importance weights for each feature map. A weighted combination of the feature maps is calculated and passed through a ReLU activation to generate the Grad-CAM heatmap. The heatmap is then resized and overlaid on the original SPECT MPI image to visualize the regions that contributed most strongly to the model's classification decision. 

Results: 
Our final ResNet18 model achieved 87.5 percent test accuracy. This was slightly higher than the custom CNN baseline, which achieved 84.4 percent accuracy, so we selected ResNet18 as our final model.
Looking at the confusion matrix on the left, the model correctly classified 24 out of 25 abnormal cases. This gives an abnormal recall of 96 percent, which is strong performance for detecting abnormal scans.
However, the model performed less well on normal cases. It correctly classified 4 out of 7 normal cases, giving a normal recall of 57.1 percent. This means the model tended to over-predict the abnormal class. In other words, some normal scans were incorrectly classified as abnormal.
The performance table shows the same pattern. The abnormal class had a higher F1-score of 0.923, while the normal class had a lower F1-score of 0.667. The weighted F1-score was 0.867, which is close to the overall accuracy and reflects the stronger performance on the larger abnormal class.
The training curve also suggests some overfitting. Training accuracy became very high, while validation accuracy fluctuated. This likely happened because the dataset was small and imbalanced.

GRAD-CAM Results:
Heatmaps concentrated primarily on myocardial perfusion regions.
The network learned clinically relevant image features rather than background information.
Grad-CAM provided visual evidence supporting model predictions.
Improved transparency of the classification pipeline.
While Grad-CAM does not directly localize ischemia, it provides interpretability by identifying image regions that most influenced model predictions

Limitations and Future Scope: 
To summarize the limitations, this project is mainly a proof-of-concept. The dataset was small and imbalanced, which likely contributed to the model performing better on abnormal cases than normal cases. Also, our model only performed binary classification, meaning it predicted normal versus abnormal, but did not separate ischemia from infarction.
Another limitation is localization. Grad-CAM helped show which image regions influenced the model’s prediction, but it only provides coarse localization, not precise vessel-level ischemia localization. We also had limited external validation, so we do not know how well the model would perform on images from other hospitals, scanners, or patient populations. Finally, we only used imaging data and did not include clinical variables such as symptoms, risk factors, ECG, or lab results.
For future work, we would use larger multi-center datasets, perform external validation, and extend the task to ischemia versus infarction classification. We could also develop vessel-specific localization, integrate clinical variables with imaging data, and improve explainability beyond Grad-CAM.



