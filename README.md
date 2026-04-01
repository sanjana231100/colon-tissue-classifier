Colon Cancer Tissue Classifier
An image classification model that identifies colon cancer tissue types from microscope images using EfficientNet-V2-S, built as part of the ML curriculum at Santa Clara University.
Overview
Each input image is just 28×28 pixels — smaller than most icons on your screen. The model classifies images into 9 tissue types including healthy tissue, tumour, and debris, which look nearly identical at this resolution.
Model Architecture

Baseline: ResNet-18
Final model: Custom classifier head on top of EfficientNet-V2-S (pretrained backbone)
Key architectural choices:

GELU activation instead of ReLU — smoother gradient flow improved learning on rare tissue types underrepresented in training data
LayerNorm before the final classification layer — stabilised training from round one by normalising the backbone's 1280 output features across wildly different scales
Data augmentation (random flips + rotations) — forced the model to learn actual tissue texture rather than relying on image position as a shortcut



Results
ModelF1 ScoreResNet-18 (baseline)0.9417EfficientNet-V2-S (ours)0.9629
+2.1 percentage points improvement over baseline across 16 training rounds.
Why F1 and Not Accuracy?
With imbalanced medical data, a model can hit 94% accuracy while completely ignoring rare classes that matter most clinically. F1 score weights all 9 tissue types equally, keeping evaluation honest.
Tech Stack

Python
PyTorch
EfficientNet-V2-S (pretrained)

Team
Built as an ML final project at Santa Clara University.
