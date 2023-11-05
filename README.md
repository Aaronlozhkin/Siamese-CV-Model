# Siamese-CV-Model

A computer vision deep learning model that learns to embed the context of an image by finetuning ResNet. Trained on the [Huggingface Snacks Dataset](https://huggingface.co/datasets/Matthijs/snacks), the Siamese CV Model will learn to distinguish ten distinct snacks from a variety of images. With this model, a similarity based reverse-image search engine is constructed.

## Installation

For a local download, utilize a [conda](conda.io) environment with [Torch](pytorch.org) and install using pip

```
pip install git+https://github.com/Aaronlozhkin/Siamese-CV-Model
```

Alternatively, open the notebook in Colab

<a target="_blank" href="https://colab.research.google.com/github/Aaronlozhkin/Siamese-CV-Model/blob/main/SiameseCVModel.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Model Architecture

This model leverages [ResNet](https://huggingface.co/docs/transformers/model_doc/resnet) and modifies its output to produce a vector representation of an image. Linear layers are then connected to output a specified embedding dimension.

### Triplet Loss

Images of snacks are loaded into Python and resized to 244x244. A siamese triplet loss is constructed based on **anchor**, **positive**, and **negative** images. Given an anchor image, the model learns to form embeddings closer to the positive image and away from the negative.

```
class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()
```

![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/2f27f72f-7126-4626-92a9-b39dc9f540b3)
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/505aeb4f-5a26-4233-bef2-2ad41dff0a75)
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/6ee9e9f9-7404-4a86-9370-b5c44c10e1d1)


## Training

```
# Hyperparameters
batch_size = 32
optimizer = optim.Adam(siamese_net.parameters(), lr=0.0001)
num_epochs = 25
```

![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/e22611e3-0232-4dd8-8b76-0c4cfcbab8b1)

Final accuracy was measured by the amount of triplets the network correctly identified as positive and negative. The final test set accuracy was 80.99%

## Cosine Similarity Ranking

The ability to represent images in a higher dimensional space based on their content opens the possibilty for a similarity ranking system. [Cosine Similarity](https://towardsdatascience.com/introduction-to-embedding-clustering-and-similarity-11dd80b00061) was utilized to achieve a similarity score between two embeddings of images. These scores were then applied on various anchor, positive, and negative triplets.

The **positive similarity score** represents the similarity between the anchor and the positive image. Likewise for the **negative similarity score**.

![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/e31cfb22-2128-4881-8c6b-76f255972a53)

\
\
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/1590c44d-7a70-4546-90c9-48200187a21f)

\
\
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/9f0f5eaf-4c97-4be7-bc7c-27dc796aa4c5)

## Reverse Image Search

A reverse image search algorithm was constructed based on the Siamese CV Model and cosine similarity. Once embeddings are calculated for every image in the dataset, a **Query Image** that is uploaded of a snack can be compared to every image in the dataset using Cosine Similarity. Then the Top-N similar images are outputted.

**Query Image:**\
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/5b30b979-8be1-4d6b-8144-8168802a4a52)

\
\
**Search Results:**\
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/c781160b-1359-44ce-a1a8-a3da810ba5e5)
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/3e0fa81b-2802-4f60-b829-9838e3c5f3f6)
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/d56b4142-5bcc-4dd5-b0d8-91ccad0016e6)
![image](https://github.com/Aaronlozhkin/Siamese-CV-Model/assets/23532191/196a5ce7-2715-4a3c-984f-351e4077bba7)





