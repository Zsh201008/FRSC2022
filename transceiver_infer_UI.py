"""
    User interface of simulating FRSC and benchmark transceiver models.
"""

import torch
import time
from tqdm import *
import matplotlib.pyplot as plt
from torchvision.transforms import *
from torch.utils.data import DataLoader
from utils.transforms import *
from speech_dataset import *
from torch.autograd import Variable
from transceiver.DeepSCS import *
from transceiver.FRSC import *
from transceiver.Traditional import *
from physical_channel import *
import numpy.random as nr
from tkinter import *
from tkinter import ttk

class TransceiverInfer():
    def __init__(self):
        self.USE_GPU=torch.cuda.is_available()
        self.DEVICE="cuda:0" if self.USE_GPU else "cpu"
        self.N_MELS=32
        self.BATCH_SIZE=128
        self.LENGTH=self.N_MELS*self.N_MELS
        self.NUM_CMDS=12
        self.CHECKTIME=0.1

        test_feature_transform = Compose([ToMelSpectrogram(n_mels=self.N_MELS), ToTensor('mel_spectrogram', 'input')])
        test_dataset = SpeechCommandsDataset("dataset_pkl\\test",Compose([LoadAudio(),FixAudioLength(),test_feature_transform]))
        self.test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False,
                                    pin_memory=self.USE_GPU, num_workers=0)

        self.model_path='pretrained_models/app_model/app_adaptive.pth'
        self.model=torch.load(self.model_path)
        self.model.to(self.DEVICE) # move model to GPU

        self.cmd_dict=self.idx2cmd()

    def idx2cmd(self):
        cmd_dict=dict()
        all_cmds='unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
        for i,cmd in enumerate(all_cmds):
            cmd_dict[i]=cmd
        return cmd_dict

    def infer(self,context_distribute='uniform',scheduler_idx=1):
        self.model.eval()
        channels=Channels()
        
        if context_distribute=='uniform':
            cmd_weight=[1 for _ in range(self.NUM_CMDS)]
        elif context_distribute=='random':
            cmd_weight=nr.random(self.NUM_CMD).tolist()
        elif context_distribute=='specified':
            cmd_weight=[0, 0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.22, 0.22, 0.22, 0.22]
        cmd_weight_frsc=torch.tensor(cmd_weight,device=DEVICE)
        scheduler=torch.load('pretrained_models/scheduler/scheduler-'+str(scheduler_idx)+'.pth')
        scheduler.to(self.DEVICE)

        system_dict={'Traditional':'',
                     'FRSC-1':'pretrained_models/FRSC_app_AWGN_1/frsc_app_awgn_1.pth',
                     'FRSC-2':'pretrained_models/FRSC_app_AWGN_2/frsc_app_awgn_2.pth',
                     'FRSC-3':'pretrained_models/FRSC_app_AWGN_3/frsc_app_awgn_3.pth',
                     'FRSC':'pretrained_models/FRSC_app_AWGN/frsc_app_awgn.pth',
                     'FRSC-Rayleigh':'pretrained_models/FRSC_app_Rayleigh/frsc_app_rayleigh.pth',
                     'FRSC-Rician':'pretrained_models/FRSC_app_Rician/frsc_app_rician.pth',
                     'FRSC-mse':'pretrained_models/FRSC_mse_AWGN/frsc_mse_awgn.pth',
                     'CNN':'pretrained_models/CNN_app_AWGN/cnn_app_awgn.pth',
                     'LSTM':'pretrained_models/LSTM_app_AWGN/lstm_app_awgn.pth',
                     'DeepSCS':'pretrained_models/DeepSCS_app_AWGN/deepscs_app_awgn.pth',
                     'DeepSCS-mse':'pretrained_models/DeepSCS_mse_AWGN/deepscs_mse_awgn.pth'}

        speed_dict={'Traditional':0,
                    'FRSC-1':0,
                    'FRSC-2':0.3,
                    'FRSC-3':0.25,
                    'FRSC':0.2,
                    'FRSC-Rayleigh':0,
                    'FRSC-Rician':0,
                    'FRSC-mse':0,
                    'CNN':0,
                    'LSTM':0,
                    'DeepSCS':0,
                    'DeepSCS-mse':0}

        channel_list=['AWGN','Rayleigh','Rician']
        snr_list=range(0,21)
        
        for system,model_path in system_dict.items():
            if system =='Traditional':
                sc_model=Traditional()
            else:
                sc_model=torch.load(model_path)
            if system!='Traditional':
                sc_model.to(self.DEVICE)

            f2=open('inference time distribution.csv','a')
            f2.write(system+',')
            for channel in channel_list:
                for snr in snr_list:
                    iteration=0
                    correct=0
                    total=0
                    infer_duration=0
                    infer_duration_list=[]

                    pbar = tqdm(self.test_loader, unit="audios", unit_scale=self.test_loader.batch_size)
                    for batch in pbar:
                        inputs = batch['input']
                        inputs = torch.unsqueeze(inputs, 1)
                        targets = batch['target']

                        inputs = Variable(inputs, requires_grad=False)
                        targets = Variable(targets, requires_grad=False)
                        inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)

                        temp_batch_size=inputs.size(0)
                        # inputs_order=[i for i in range(0,self.BATCH_SIZE)]
                        if system!='Traditional':
                            if 'FRSC' in system:
                                inputs, inputs_order, cmd_entropy=scheduler(inputs,False,cmd_weight_frsc)
                            x=inputs.reshape(temp_batch_size,1,self.N_MELS,self.N_MELS)
                            x=sc_model.speech_encoder(x)
                            x=sc_model.channel_encoder(x)
                            x=x.reshape(temp_batch_size,-1,self.LENGTH)
                        else:
                            x=inputs.reshape(-1).tolist()
                            x=sc_model.quantize(x)
                            x=sc_model.naive_encoder(x)
                            x=torch.tensor(x,dtype=torch.float32,device=self.DEVICE)
                            x=x.reshape(temp_batch_size,-1,self.LENGTH)
                        x=power_normalize(x)

                        n_var=snr2noise(snr)
                        if channel == 'AWGN':
                            x = channels.AWGN(x, n_var)
                        elif channel  == 'Rayleigh':
                            x = channels.Rayleigh(x, n_var)
                        elif channel  == 'Rician':
                            x = channels.Rician(x, n_var)
                        else:
                            print("Error: No such channel!")
                            x = None

                        if system!='Traditional':
                            x=x.reshape(temp_batch_size,-1,self.N_MELS,self.N_MELS)
                            x=sc_model.channel_decoder(x)
                            x=sc_model.speech_decoder(x)
                            x=sc_model.last_layer(x)
                        else:
                            x=x.reshape(-1).tolist()
                            x=[round(i) for i in x]
                            x=sc_model.naive_decoder(x)
                            x=sc_model.dequantize(x)
                            x=torch.tensor(x,device=self.DEVICE)
                            x=x.reshape(temp_batch_size,-1,self.N_MELS,self.N_MELS)

                        start_infer=time.time()
                        outputs, finish_layers, uncertainty=self.model(x,True,speed_dict[system],-1,targets)
                        end_infer=time.time()
                        infer_duration_list.append(end_infer-start_infer)
                        infer_duration+=(end_infer-start_infer)

                        f2.write(str(end_infer-start_infer)+',')

                        _,top_index = outputs.topk(1)
                        total+=targets.size(0)
                        correct+=(top_index.view(-1)==targets).sum().item()

                        pbar.set_postfix({
                            'acc': "%.02f%%" % (100*correct/total)
                        })

                        iteration+=1
                        
                    accuracy = correct*1.0/total
                    infer_duration=infer_duration/iteration/self.BATCH_SIZE
                    f1=open('accuracy-inference speed.csv','a')
                    f1.write(str(system)+','+str(channel)+','+str(snr)+',0,'+str(accuracy)+','+str(infer_duration)+'\n')
                    f1.close()
            
            f2.write('\n')
            f2.close()
            del sc_model

if __name__=='__main__':
    transceiver_infer=TransceiverInfer()
    transceiver_infer.infer()

    cmds=['on','off','yes','no','stop','go','left','right','up','down']

    root=Tk()
    root.title('Simulation of Speech Command Recognition')
    frm = ttk.Frame(root,borderwidth=50)
    frm.grid()

    cmd_weight_label=ttk.Label(frm,text='command weight configuration',font=('Arial',11)).grid(column=0,columnspan=10,row=0,pady=5)
    cmd_weight_entry_list=[]
    for i,cmd in enumerate(cmds):
        ttk.Label(frm,text=cmd,font=('Arial',9)).grid(column=i,row=1)
    for i in range(len(cmds)):
        cmd_weight_entry_list.append(ttk.Entry(frm, width=5).grid(column=i,row=2,pady=5))

    batch_size_label=ttk.Label(frm,text='batch size',font=('Arial',11)).grid(column=0,columnspan=3,row=3,pady=5)
    scheduler_label=ttk.Label(frm,text='FRSC scheduler',font=('Arial',11)).grid(column=3,columnspan=4,row=3,pady=5)
    checkpoint_label=ttk.Label(frm,text='checkpoint',font=('Arial',11)).grid(column=7,columnspan=3,row=3,pady=5)
    batch_size_entry=ttk.Entry(frm,width=15).grid(column=0,columnspan=3,row=4)

    scheduler=StringVar()
    scheduler.set('No scheduler')
    scheduler_list=['No scheduler','Scheduler-1','Scheduler-2']
    scheduler_cbox=ttk.Combobox(frm,width=20,state='readonly',cursor='arrow',value=scheduler_list).grid(column=3,columnspan=4,row=4)
    checkpoint_entry=ttk.Entry(frm,width=15).grid(column=7,columnspan=3,row=4)

    sequence_label=ttk.Label(frm,text='sending command sequence',font=('Arial',11)).grid(column=0,columnspan=10,row=5,pady=5)
    sequence_entry=ttk.Entry(frm,width=70).grid(column=0,columnspan=10,row=6)

    button=ttk.Button(frm,text='transmit',command=root.destroy).grid(column=0,columnspan=10,row=7,pady=10)

    # progress_label=ttk.Label(frm,text='...... TRANSMITTING ......',font=('Arial',11)).grid(column=0,columnspan=10,row=8,pady=10)
    batch_size=64
    num_audio_trans=IntVar()
    num_audio_trans.set(64)
    progress_bar=ttk.Progressbar(frm,length=490,maximum=batch_size,variable=num_audio_trans).grid(column=0,columnspan=10,row=8,pady=10)

    traditional_label=ttk.Label(frm,text='************************************ FRSC ************************************',font=('Arial',11)).grid(column=0,columnspan=10,row=9,pady=3)
    traditional_result=ttk.Label(frm,text='on, off, stop, go, -, -, -, -').grid(column=0,columnspan=10,row=10,pady=3)

    traditional_label=ttk.Label(frm,text='********************************** DeepSC-S **********************************',font=('Arial',11)).grid(column=0,columnspan=10,row=11,pady=3)
    traditional_result=ttk.Label(frm,text='on, off, stop, go, -, -, -, -').grid(column=0,columnspan=10,row=12,pady=3)

    traditional_label=ttk.Label(frm,text='************************************ CNN *************************************',font=('Arial',11)).grid(column=0,columnspan=10,row=13,pady=3)
    traditional_result=ttk.Label(frm,text='on, off, stop, go, -, -, -, -').grid(column=0,columnspan=10,row=14,pady=3)

    traditional_label=ttk.Label(frm,text='************************************ LSTM ************************************',font=('Arial',11)).grid(column=0,columnspan=10,row=15,pady=3)
    traditional_result=ttk.Label(frm,text='on, off, stop, go, -, -, -, -').grid(column=0,columnspan=10,row=16,pady=3)

    traditional_label=ttk.Label(frm,text='************************ traditional telephone system ************************',font=('Arial',11)).grid(column=0,columnspan=10,row=17,pady=3)
    traditional_result=ttk.Label(frm,text='on, off, stop, go, -, -, -, -').grid(column=0,columnspan=10,row=18,pady=3)

    root.mainloop()
