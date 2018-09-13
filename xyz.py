class Darknet(nn.Module):
    
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = Create_modules(self.blocks)
        self.header = torch.IntTensor([0,0,0,0])
        self.seen = 0

        
        
    def get_blocks(self):
        return self.blocks

    
    def get_module_list(self):
        return self.module_list
    
    
    
    def forward(self, x, CUDA = False):
        
        
        detections=[]
        
        modules = self.blocks[1:]
        outputs={}
        
        
        write=0
    
        
        for i in range(len(modules)):
            #print(len(modules))
            modules_type = (modules[i]['type'])
        
            if modules_type == 'convolution' or modules_type == 'upsample' or modules_type == 'maxpool' :
                x=self.modules_list[i](x)
                print(x)
                outputs[i]=x
                print(outputs[i])
            elif modules_type == 'route':
                layers = modules[i]['layers']
                layers = [int(a) for a in layers]
                
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                    
                if len(layers) == 1:
                    x =  outputs[i + (layers[0])]
                    
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs [i + layers[0]]
                    map2 = outputs [i + layers[1]]
                    
                    x = torch.cat((map1, map2), 1)
                
                outputs[i] = x
                
            elif modules_type == 'shortcut':
                from_ = int(modules[i]['from'])
                print(from_)
                print(i)
                print(outputs[i-1])
                x = outputs[i-1] + outputs[i + from_]
                outputs[i] = x
                
            
            elif modules_type == 'yolo':
                
                anchors=self.module_list[i][0].anchors
                
                inp_dim=int(self.net_info['height'])
                
                num_classes=int(module[i]['classes'])
                
                x=x.data
                x=predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
                if type(x) == int:
                    continue
                    
                if not write:
                    detections = x
                    write = 1 
                    
                else:
                    detections = torch.cat((detection, x), 1)
                    
                outputs[i] = outputs[i-1]
                
                
                
        try:
            return detections
        
        except:
            return 0
        
                 
    def load_weights(self, weightfile):
        
        fp = open(weightfile, 'rb')
        
        
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
            
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        
        
        
        for i in range(len(self.module_list)):
            
            module_type = self.blocks[i+1]['type']
            
            if module_type == 'convolutional':
                model = self.module_list[i]
                
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                    
                except:
                    batch_normalize = 0
                    
                
                conv = model[0]
                
                if batch_normalize:
                    
                    bn = model[1]
                    
                    num_bn_biases = bn.bias.numel()
                    
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    
                    
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                    
                    
                else:
                    num_biases = conv.bias.numel()
                    
                    conv_biases = torch.from_numpy(weights[ptr:ptr  + num_biases])
                    ptr += num_biases
                    
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    conv.bias.data.copy_(conv_biases)
                    
                
                num_weights = conv.weight.numel()
                
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights
                #print(conv_weights.shape)
                conv_weights = conv_weights.view_as(conv.weight.data)
                
                conv.weight.data.copy_(conv_weights)
                
                
                    
    def save_weights(self, savedfile, cutoff = 0):
        
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
            
        fp = open(savedfile, 'wb')
        
        self.header[3] = self.seen
        header = self.header
        
        header = header.numpy()
        header.tofile(fp)
        
        
        for i in range(len(self.module_list)):
            module_type = self.block[i+1]['type']
            
            
            if module_type == 'convolutional':
                model = self.module_list[i]
                
                try:
                    batch_normalize = int(self.block[i+1]['batch_normalize'])
                    
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if batch_normalize:
                    bn = model[1]
                    
                    
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weights.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)
                    
                else:
                    cpu(conv.bias.data).numpy().tofile(fp)
                    
                
                cpu(conv.weights.data).numpy().tofile(fp)
                
                
            
