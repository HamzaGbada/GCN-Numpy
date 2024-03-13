from GCN_scratch.layers import GCNLayer


class GCN:

    def __init__(self, in_feat, hid_feat, out_feat, layers=2):
        self.init_layer = GCNLayer(in_feat, hid_feat)
        self.hidden_layer = GCNLayer(hid_feat, hid_feat)
        self.out_layer = GCNLayer(hid_feat, out_feat)

    def forword(self):
        pass