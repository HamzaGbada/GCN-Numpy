from main import gcn_model, A_Laplacien, X, labels
from src.optimizer import Gradient_Descent_Optimizer
from src.layers import Utils
import numpy as np

train_nodes = np.array([0, 1, 8])
test_nodes = np.array([i for i in range(labels.shape[0]) if i not in train_nodes])
opt2 = Gradient_Descent_Optimizer(alpha=2e-2, w=2.5e-2)

embeds = list()
accuacy = list()
train_losses = list()
test_losses = list()

loss_min = 1e6
es_iters = 0
early_stop = 50
epochs = 500

for epoch in range(epochs):

    y_pred = gcn_model.forward(A_Laplacien, X)

    opt2(y_pred, labels, train_nodes)

    for layer in reversed(gcn_model.layers):
        layer.backward(opt2, update=True)

    embeds.append(gcn_model.embedding(A_Laplacien, X))
    acc = (np.argmax(y_pred, axis=1) == np.argmax(labels, axis=1))[
        [i for i in range(labels.shape[0]) if i not in train_nodes]
    ]
    accuacy.append(acc.mean())

    loss = Utils.xent(y_pred, labels)
    loss_train = loss[train_nodes].mean()
    loss_test = loss[test_nodes].mean()

    train_losses.append(loss_train)
    test_losses.append(loss_test)

    if loss_test < loss_min:
        loss_min = loss_test
        early_stop_iters = 0
    else:
        early_stop_iters += 1

    if es_iters > early_stop:
        print("Early Stop")
        break
    print(f"CurrentEpoch: {epoch + 1}, Training Loss: {loss_train:.5f}, Testing Loss: {loss_test:.5f}")

train_losses = np.array(train_losses)
test_losses = np.array(test_losses)
