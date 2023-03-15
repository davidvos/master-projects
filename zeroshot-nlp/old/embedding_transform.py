import torch
import os
import pickle
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingTransform():

    def __init__(self, transform_learning_method, source_embeddings, target_embeddings, str_tok, end_tok):
        self.transform_learning_method = transform_learning_method
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.learn_transform()
        self.str_tok = str_tok
        self.end_tok = end_tok

    def learn_transform(self):
        if self.transform_learning_method == "SGD":
            self.transform = self.learn_transform_SGD().to(DEVICE)
        elif self.transform_learning_method == "SVD":
            self.transform = self.learn_transform_SVD().to(DEVICE)
        elif self.transform_learning_method == "CCA":
            pass
        else:
            raise ValueError(f"Unknown transform learning method: {self.transform_learning_method}")

    def english_to_dutch(self, english_embedding):
        pass

    def dutch_to_english(self, dutch_embedding, normalize=False):
        """
        Transform a dutch sentence to an embedded english sentence
        """
        transformed_sentence = dutch_embedding @ self.transform
        if normalize:
            norms = torch.norm(transformed_sentence, dim=1)
            transformed_sentence = transformed_sentence / norms.view(-1, 1)
        return torch.cat((self.str_tok.view(1, 768), transformed_sentence, self.end_tok.view(1, 768)))

    def learn_transform_SGD(self, store_model_fn="./models/sgd_transform.pkl", load_model=False):
        """
        Learn a projection from the source to the target embedding space using SGD.
        """

        if load_model and os.path.exists(store_model_fn):
            print(f"Loading SGD trained transform from disk: {store_model_fn}")
            with open(store_model_fn, "rb") as f:
                data = pickle.load(f)
                transform = data["transform"]
        else:
            transform = torch.zeros((256, 768), requires_grad=True, device=DEVICE)
            torch.nn.init.xavier_uniform_(transform)
            optim = torch.optim.Adam((transform,), lr=0.001)
            loss_fn = torch.nn.MSELoss().to(DEVICE)
            
            targets = self.target_embeddings.clone().to(DEVICE)
            samples = self.source_embeddings.clone().to(DEVICE)

            batch_size = 32

            losses = []
            for ep in range(1000):

                shf = torch.randperm(samples.shape[0])

                samples = samples[shf]
                targets = targets[shf]

                tot_loss = 0
                for b_start in range(0, samples.shape[0], batch_size):

                    optim.zero_grad()

                    batch_inputs = samples[b_start:b_start+batch_size, :].clone().detach()
                    batch_targets = targets[b_start:b_start+batch_size, :].clone().detach()

                    batch_outputs = batch_inputs @ transform

                    loss = loss_fn(batch_outputs, batch_targets)
                    loss.backward()
                    optim.step()

                    tot_loss += loss.item()
                
                losses.append(tot_loss)

                if len(losses) > 10 and np.mean(losses[-10:]) < np.mean(losses[-4:]):
                    print("early stopping condition reached")
                    break
                print(f"epoch: {ep} loss: {tot_loss}")
            
            # store the trained model on disk
            with open(store_model_fn, "wb") as f:
                pickle.dump({"transform":transform}, f)
        return transform

    def learn_transform_SVD(self):
        """
        Learn an orthogonal projection from the source embedding space to the target embedding space
        using SVD.
        """
        XX = self.source_embeddings.T @ self.target_embeddings
        XX_ = self.target_embeddings.T @ self.source_embeddings
        U, _, V = torch.svd(XX)
        U_, _, V_ = torch.svd(XX_)
        # transform_inverse = V @ U.T
        transform = V_ @ U_.T
        return transform