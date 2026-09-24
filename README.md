# Classifying Respiratory Diseases from X-rays

<ul>
    <li><a href="#what-is-chest-disease-detection">What is Chest Disease Detection by X-ray?</a></li>
    <li><a href="#datasets-used">Datasets Used in This Project</a></li>
    <li><a href="#why-use-xrays">Why We Use X-rays Instead of PCR?</a></li>
    <li><a href="#class-distribution">Class Distribution</a></li>
    <li><a href="#imbalanced-data">Handling Class Imbalance</a></li>
    <li><a href="#clahe">Image Preprocessing — CLAHE</a></li>
    <li><a href="#vgg19">Modeling</a></li>
    <li><a href="#model-evaluation">Model Evaluation</a></li>
    <li><a href="#grad-cam-visualizations">Grad-CAM Visualization</a></li>
    <li><a href="#deployment">Deployment</a></li>
    <li><a href="#authors">Authors</a></li>
</ul>

---

<h2 id="what-is-chest-disease-detection">What is Chest Disease Detection by X-ray?</h2>

<p>
Chest disease detection using X-rays applies deep learning techniques to analyze
chest radiographs and identify patterns associated with respiratory conditions.
In this project, a multi-class classification system was developed to classify
chest X-ray images into six categories.
</p>

<ul>
    <li><strong>Normal</strong></li>
    <li><strong>Pneumonia — Bacterial</strong></li>
    <li><strong>Pneumonia — Viral</strong></li>
    <li><strong>Tuberculosis</strong></li>
    <li><strong>Lung Opacity</strong></li>
    <li><strong>COVID-19</strong></li>
</ul>

<p>
The project combines image preprocessing, data augmentation, class weighting,
and transfer learning using VGG19.
</p>

---

<h2 id="datasets-used">Datasets Used in This Project</h2>

<p>
Three public chest X-ray datasets were combined to build the final classification dataset.
</p>

<h3>Dataset 1 — COVID-19 Radiography Database</h3>

<table>
    <tr>
        <th>Class</th>
        <th>Images</th>
    </tr>
    <tr><td>Normal</td><td>10,200</td></tr>
    <tr><td>Viral Pneumonia</td><td>1,345</td></tr>
    <tr><td>COVID-19</td><td>3,616</td></tr>
    <tr><td>Lung Opacity</td><td>6,012</td></tr>
    <tr><td>Bacterial Pneumonia</td><td>0</td></tr>
    <tr><td>Tuberculosis</td><td>0</td></tr>
</table>

<h3>Dataset 2 — Curated Chest X-Ray Image Dataset for COVID-19</h3>

<table>
    <tr>
        <th>Class</th>
        <th>Images</th>
    </tr>
    <tr><td>Normal</td><td>3,270</td></tr>
    <tr><td>Viral Pneumonia</td><td>1,656</td></tr>
    <tr><td>COVID-19</td><td>1,281</td></tr>
    <tr><td>Lung Opacity</td><td>0</td></tr>
    <tr><td>Bacterial Pneumonia</td><td>3,001</td></tr>
    <tr><td>Tuberculosis</td><td>0</td></tr>
</table>

<h3>Dataset 3 — Tuberculosis X-Ray Dataset</h3>

<table>
    <tr>
        <th>Class</th>
        <th>Images</th>
    </tr>
    <tr><td>Normal</td><td>514</td></tr>
    <tr><td>Viral Pneumonia</td><td>0</td></tr>
    <tr><td>COVID-19</td><td>0</td></tr>
    <tr><td>Lung Opacity</td><td>0</td></tr>
    <tr><td>Bacterial Pneumonia</td><td>0</td></tr>
    <tr><td>Tuberculosis</td><td>2,494</td></tr>
</table>

---

<h2 id="why-use-xrays">Why We Use X-rays Instead of PCR?</h2>

<p>
X-rays provide a quick, non-invasive, and widely available method for visualizing
the lungs. While PCR can detect specific pathogens or genetic material, chest
X-rays provide information about structural abnormalities such as opacities
and other changes in the lungs.
</p>

<p>
For this project, X-rays were therefore used as the primary imaging modality
for multi-class respiratory disease classification.
</p>

---

<h2 id="class-distribution">Class Distribution</h2>

<p>
The final dataset contains <strong>25,905 images</strong> distributed across six classes.
</p>

<table>
    <tr>
        <th>Class</th>
        <th>Number of Images</th>
        <th>Percentage</th>
    </tr>
    <tr>
        <td>Normal</td>
        <td>6,500</td>
        <td>25.1%</td>
    </tr>
    <tr>
        <td>Lung Opacity</td>
        <td>6,012</td>
        <td>23.2%</td>
    </tr>
    <tr>
        <td>COVID-19</td>
        <td>4,897</td>
        <td>18.9%</td>
    </tr>
    <tr>
        <td>Pneumonia-Bacterial</td>
        <td>3,001</td>
        <td>11.6%</td>
    </tr>
    <tr>
        <td>Pneumonia-Viral</td>
        <td>3,001</td>
        <td>11.6%</td>
    </tr>
    <tr>
        <td>Tuberculosis</td>
        <td>2,494</td>
        <td>9.6%</td>
    </tr>
    <tr>
        <th>Total</th>
        <th>25,905</th>
        <th>100%</th>
    </tr>
</table>

<img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Dataset.png"
     alt="Dataset Distribution">

---

<h2 id="imbalanced-data">Handling Class Imbalance</h2>

<p>
The dataset contains different numbers of images across the classes.
Class imbalance can cause a model to favor majority classes and perform
poorly on underrepresented classes.
</p>

<p>
Two approaches were used to address this issue:
</p>

<h3>Data Augmentation</h3>

<p>
Data augmentation applies controlled transformations to training images
to increase data diversity and improve model generalization.
</p>

<h3>Class Weights</h3>

<p>
Class weights assign higher importance to underrepresented classes during
training, increasing the penalty for misclassifying minority-class samples.
</p>

<pre>
from sklearn.utils.class_weight import compute_class_weight

class_indices = train_generator.class_indices
class_labels = list(class_indices.keys())

train_labels = train_generator.classes

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights_dict = dict(enumerate(class_weights))

print("Class Labels:", class_labels)
print("Class Weights:", class_weights_dict)
</pre>

---

<h2 id="clahe">Image Preprocessing — CLAHE</h2>

<p>
<strong>CLAHE (Contrast-Limited Adaptive Histogram Equalization)</strong>
was used to improve local contrast in chest X-ray images.
</p>

<p>
The configuration used in the project was:
</p>

<pre>
clipLimit = 2.0
tileGridSize = (8, 8)
</pre>

<p>
CLAHE performs localized histogram equalization while limiting excessive
contrast amplification, making it suitable for enhancing subtle structures
in medical images.
</p>

<h3>CLAHE Comparison</h3>

<table>
    <tr>
        <th>Condition</th>
        <th>Original / CLAHE Comparison</th>
    </tr>
    <tr>
        <td><strong>COVID-19</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_COVID-19.png"
                 alt="COVID-19 CLAHE">
        </td>
    </tr>
    <tr>
        <td><strong>Normal</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Normal.png"
                 alt="Normal CLAHE">
        </td>
    </tr>
    <tr>
        <td><strong>Lung Opacity</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Lung_Opacity.png"
                 alt="Lung Opacity CLAHE">
        </td>
    </tr>
    <tr>
        <td><strong>Pneumonia-Bacterial</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Pneumonia-Bacterial.png"
                 alt="Bacterial Pneumonia CLAHE">
        </td>
    </tr>
    <tr>
        <td><strong>Pneumonia-Viral</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Pneumonia-Viral.png"
                 alt="Viral Pneumonia CLAHE">
        </td>
    </tr>
    <tr>
        <td><strong>Tuberculosis</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/CLAHE_Tuberculosis.png"
                 alt="Tuberculosis CLAHE">
        </td>
    </tr>
</table>

---

<h2 id="vgg19">Modeling</h2>

<h3>VGG19</h3>

<p>
<strong>VGG19</strong> is a convolutional neural network developed by
the Visual Geometry Group at Oxford University. The model was originally
pre-trained on ImageNet and was adapted to the six-class chest X-ray
classification task using transfer learning.
</p>

<p>
The project compares four approaches:
</p>

<ul>
    <li>VGG19</li>
    <li>VGG19 with CLAHE</li>
    <li>VGG16</li>
    <li>CNN trained from scratch</li>
</ul>

<h3>Model Comparison</h3>

<table>
    <tr>
        <th>Model</th>
        <th>Train Loss</th>
        <th>Val Loss</th>
        <th>Test Loss</th>
        <th>Train Accuracy</th>
        <th>Val Accuracy</th>
        <th>Test Accuracy</th>
    </tr>
    <tr>
        <td><strong>VGG19 with CLAHE</strong></td>
        <td><strong>0.0681</strong></td>
        <td><strong>0.2247</strong></td>
        <td><strong>0.2257</strong></td>
        <td><strong>97.46%</strong></td>
        <td><strong>92.24%</strong></td>
        <td><strong>92.56%</strong></td>
    </tr>
    <tr>
        <td><strong>VGG19</strong></td>
        <td>0.1166</td>
        <td>0.2392</td>
        <td>0.2365</td>
        <td>95.54%</td>
        <td>91.63%</td>
        <td>91.65%</td>
    </tr>
    <tr>
        <td><strong>From Scratch</strong></td>
        <td>0.9191</td>
        <td>0.8634</td>
        <td>0.8587</td>
        <td>91.91%</td>
        <td>86.34%</td>
        <td>85.87%</td>
    </tr>
    <tr>
        <td><strong>VGG16</strong></td>
        <td>0.4927</td>
        <td>0.5423</td>
        <td>0.5021</td>
        <td>80.02%</td>
        <td>77.61%</td>
        <td>79.89%</td>
    </tr>
</table>

<p>
Among the evaluated approaches, <strong>VGG19 with CLAHE</strong> achieved
the highest test accuracy at <strong>92.56%</strong>.
</p>

<h3>Training Performance</h3>

<h4>Accuracy</h4>

<img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/TRAIN-VGG19.png"
     alt="VGG19 Accuracy">

<h4>Loss</h4>

<img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Loss_VGG19.png"
     alt="VGG19 Loss">

---

<h2 id="model-evaluation">Model Evaluation</h2>

<p>
The final VGG19 model was evaluated using Precision, Recall, F1-score,
Accuracy, and a Confusion Matrix.
</p>

<h3>Classification Report</h3>

<table>
    <tr>
        <th>Class</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1-Score</th>
        <th>Support</th>
    </tr>
    <tr><td>0.0</td><td>0.98</td><td>0.97</td><td>0.98</td><td>491</td></tr>
    <tr><td>1.0</td><td>0.96</td><td>0.89</td><td>0.93</td><td>602</td></tr>
    <tr><td>2.0</td><td>0.90</td><td>0.96</td><td>0.93</td><td>650</td></tr>
    <tr><td>3.0</td><td>0.86</td><td>0.87</td><td>0.87</td><td>301</td></tr>
    <tr><td>4.0</td><td>0.83</td><td>0.86</td><td>0.84</td><td>301</td></tr>
    <tr><td>5.0</td><td>1.00</td><td>0.99</td><td>0.99</td><td>250</td></tr>
    <tr>
        <th>Accuracy</th>
        <td></td>
        <td></td>
        <th>0.93</th>
        <th>2595</th>
    </tr>
    <tr>
        <th>Macro Avg</th>
        <td>0.92</td>
        <td>0.92</td>
        <td>0.92</td>
        <td>2595</td>
    </tr>
    <tr>
        <th>Weighted Avg</th>
        <td>0.93</td>
        <td>0.93</td>
        <td>0.93</td>
        <td>2595</td>
    </tr>
</table>

<ul>
    <li><strong>Precision:</strong> The proportion of predicted positive samples that were correctly classified.</li>
    <li><strong>Recall:</strong> The proportion of actual positive samples that were correctly identified.</li>
    <li><strong>F1-Score:</strong> The harmonic mean of Precision and Recall.</li>
    <li><strong>Support:</strong> The number of actual samples belonging to each class.</li>
</ul>

<h3>Confusion Matrix</h3>

<img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Confusion%20Matrix_VGG19.png"
     alt="Confusion Matrix">

---

<h2 id="grad-cam-visualizations">Grad-CAM Visualization for X-Ray Classes</h2>

<p>
<strong>Grad-CAM (Gradient-weighted Class Activation Mapping)</strong>
was used to visualize the image regions that contributed to the model's
prediction.
</p>

<p>
This provides an additional interpretability layer for examining the
model's attention across different X-ray classes.
</p>

<table>
    <tr>
        <th>Class</th>
        <th>Grad-CAM Visualization</th>
    </tr>
    <tr>
        <td><strong>Normal</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Normal_GradCAM.png"
                 alt="Normal GradCAM">
        </td>
    </tr>
    <tr>
        <td><strong>Pneumonia-Bacterial</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Pneumonia-Bacterial_GradCAM.png"
                 alt="Bacterial Pneumonia GradCAM">
        </td>
    </tr>
    <tr>
        <td><strong>COVID-19</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/COVID-19_GradCAM.png"
                 alt="COVID-19 GradCAM">
        </td>
    </tr>
    <tr>
        <td><strong>Pneumonia-Viral</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Pneumonia-Viral_GradCAM.png"
                 alt="Viral Pneumonia GradCAM">
        </td>
    </tr>
    <tr>
        <td><strong>Tuberculosis</strong></td>
        <td>
            <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/IMAGE%20FOR%20MODEL/Tuberculosis_GradCAM.png"
                 alt="Tuberculosis GradCAM">
        </td>
    </tr>
</table>

---

<h2 id="deployment">Deployment Website</h2>

<p>
Explore the deployed version of our Chest Disease Detection by X-ray
web application:
</p>

<p>
<a href="https://github.com/fatma2123456/BreatheAI-Website/blob/main/README.md">
BreatheAI Website
</a>
</p>

---

<h2 id="authors">Authors</h2>

<pre>
Karim Ehab - Computer Engineering - <b><a href="https://github.com/Karim-Ehab-AI">Karim-Ehab-AI</a></b>
Fatma Elzhra ahmed - Artificial Intelligence Engineering - <b><a href="https://github.com/fatma2123456">fatma2123456</a></b>
Hanin Mustafa - Computer Systems Engineering  - <b><a href="https://github.com/HaninMustafa9">HaninMustafa9</a></b>
Abdelrahman Mohamed - Computer Science Engineering - <b><a href="https://github.com/AbdelrahmanMohamed252">AbdelrahmanMohamed252</a></b>
Mahmoud Anas - Computer Science - <b><a href="https://github.com/MahmoudAnas046">MahmoudAnas046</a></b>
 
Supervised By :
Eng / Mahmoud Talaat 
Ai Engineer at MCiT (Ministry of Communication and Information Technology)
TA at Zewail University ( Artificial intelligence and Data Science )
  <div style="text-align: right;">
                                                                                                     <img src="https://github.com/fatma2123456/CLASSIFYING-RESPIRATORY-DISEASES-FROM-X-RAYS/blob/main/Image/Picture2_20241025_195821_0000.png" alt="Government Logo" width="150"/> 
</div>
</pre>

---

<h2>Disclaimer</h2>

<p>
This project is intended for <strong>educational and research purposes only</strong>.
Model predictions should not be considered a medical diagnosis or used as
a replacement for evaluation by a qualified healthcare professional.
</p>
