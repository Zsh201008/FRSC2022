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
import tkinter as tk
from tkinter import ttk

class TransceiverInferUI():
    def __init__(self):
        self.USE_GPU=torch.cuda.is_available()
        self.DEVICE="cuda:0" if self.USE_GPU else "cpu"
        self.N_MELS=32
        self.BATCH_SIZE=32
        self.LENGTH=self.N_MELS*self.N_MELS
        self.NUM_CMDS=12

        test_feature_transform = Compose([ToMelSpectrogram(n_mels=self.N_MELS), ToTensor('mel_spectrogram', 'input')])
        test_dataset = SpeechCommandsDataset("dataset_demo",Compose([LoadAudio(),FixAudioLength(),test_feature_transform]))
        self.test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False,
                                    pin_memory=self.USE_GPU, num_workers=0)

        self.app_model=torch.load('pretrained_models/app_model/app_adaptive.pth')
        self.app_model.to(self.DEVICE)
        self.app_model.eval()

        self.channels=Channels()

        self.cmds='unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
        self.cmd_dict=self.idx2cmd()

        self.system_dict={'Traditional':'',
                          'FRSC':'pretrained_models/FRSC_app_AWGN/frsc_app_awgn.pth',
                          'CNN':'pretrained_models/CNN_app_AWGN/cnn_app_awgn.pth',
                          'LSTM':'pretrained_models/LSTM_app_AWGN/lstm_app_awgn.pth',
                          'DeepSCS':'pretrained_models/DeepSCS_app_AWGN/deepscs_app_awgn.pth'}

        # self.speed_dict={'Traditional':0,
        #                  'FRSC':0,
        #                  'CNN':0,
        #                  'LSTM':0,
        #                  'DeepSCS':0}

        self.root=tk.Tk()
        self.root.title('Simulation of Speech Command Recognition')
        self.frm = ttk.Frame(self.root,borderwidth=50)
        self.frm.grid()

        ttk.Label(self.frm,text='command weight configuration',font=('Arial',11)).grid(column=0,columnspan=10,row=0,pady=5)
        self.cmd_weight_entry_list=[]
        for i,cmd in enumerate(self.cmds[2:]):
            ttk.Label(self.frm,text=cmd,font=('Arial',9)).grid(column=i,row=1)
        for i in range(len(self.cmds)-2):
            cmd_weight_entry=ttk.Entry(self.frm, width=5)
            cmd_weight_entry.grid(column=i,row=2,pady=5)
            self.cmd_weight_entry_list.append(cmd_weight_entry)

        ttk.Label(self.frm,text='batch size',font=('Arial',11)).grid(column=0,columnspan=3,row=3,pady=5)
        ttk.Label(self.frm,text='FRSC scheduler',font=('Arial',11)).grid(column=3,columnspan=4,row=3,pady=5)
        ttk.Label(self.frm,text='checkpoint',font=('Arial',11)).grid(column=7,columnspan=3,row=3,pady=5)

        self.batch_size_entry=tk.Entry(self.frm,width=15)
        self.batch_size_entry.grid(column=0,columnspan=3,row=4)
        self.scheduler=tk.StringVar()
        self.scheduler.set('No scheduler')
        self.scheduler_list=['No scheduler','Scheduler-1','Scheduler-2']
        self.scheduler_cbox=ttk.Combobox(self.frm,width=20,state='readonly',cursor='arrow',value=self.scheduler_list)
        self.scheduler_cbox.grid(column=3,columnspan=4,row=4)
        self.checkpoint_entry=tk.Entry(self.frm,width=15)
        self.checkpoint_entry.grid(column=7,columnspan=3,row=4)

        for batch in self.test_loader:
            self.inputs=batch['input']
            self.targets=batch['target']
            target_list=batch['target'].tolist()
        
        is_first=True
        self.transmit_sequence=''
        for target in target_list:
            if target==1:
                continue
            cmd=self.cmd_dict[target]
            if cmd=='unknown':
                if is_first:
                    cmd='bird'
                    is_first=False
                else:
                    cmd='cat'
            self.transmit_sequence=self.transmit_sequence+cmd+', '
        
        ttk.Label(self.frm,text='sending command sequence',font=('Arial',11)).grid(column=0,columnspan=10,row=5,pady=5)
        ttk.Label(self.frm,text=self.transmit_sequence).grid(column=0,columnspan=10,row=6)

        self.total_systems=len(self.system_dict.keys())
        self.num_systems=tk.IntVar()
        self.num_systems.set(0)
        self.transmit_button=ttk.Button(self.frm,text='transmit',command=self.transmit)
        self.transmit_button.grid(column=0,columnspan=10,row=7,pady=10)
        self.progress_bar=ttk.Progressbar(self.frm,length=490,maximum=self.total_systems,variable=self.num_systems)
        self.progress_bar.grid(column=0,columnspan=10,row=8,pady=10)

        ttk.Label(self.frm,text='************************************ FRSC ************************************',font=('Arial',11)).grid(column=0,columnspan=10,row=9,pady=3)
        self.frsc_infer=tk.StringVar()
        self.frsc_infer.set(' ')
        ttk.Label(self.frm,textvariable=self.frsc_infer).grid(column=0,columnspan=10,row=10,pady=3)

        ttk.Label(self.frm,text='********************************** DeepSC-S **********************************',font=('Arial',11)).grid(column=0,columnspan=10,row=11,pady=3)
        self.deepscs_infer=tk.StringVar()
        self.deepscs_infer.set(' ')
        ttk.Label(self.frm,textvariable=self.deepscs_infer).grid(column=0,columnspan=10,row=12,pady=3)

        ttk.Label(self.frm,text='************************************ CNN *************************************',font=('Arial',11)).grid(column=0,columnspan=10,row=13,pady=3)
        self.cnn_infer=tk.StringVar()
        self.cnn_infer.set(' ')
        ttk.Label(self.frm,textvariable=self.cnn_infer).grid(column=0,columnspan=10,row=14,pady=3)

        ttk.Label(self.frm,text='************************************ LSTM ************************************',font=('Arial',11)).grid(column=0,columnspan=10,row=15,pady=3)
        self.lstm_infer=tk.StringVar()
        self.lstm_infer.set(' ')
        ttk.Label(self.frm,textvariable=self.lstm_infer).grid(column=0,columnspan=10,row=16,pady=3)

        ttk.Label(self.frm,text='************************ traditional telephone system ************************',font=('Arial',11)).grid(column=0,columnspan=10,row=17,pady=3)
        self.tradi_infer=tk.StringVar()
        self.tradi_infer.set(' ')
        ttk.Label(self.frm,textvariable=self.tradi_infer).grid(column=0,columnspan=10,row=18,pady=3)

    def idx2cmd(self):
        cmd_dict=dict()
        for i,cmd in enumerate(self.cmds):
            cmd_dict[i]=cmd
        return cmd_dict

    def transmit(self):
        cmd_weight=[]
        for cmd_weight_entry in self.cmd_weight_entry_list:
            cmd_weight.append(float(cmd_weight_entry.get()))

        scheduler_content=self.scheduler_cbox.get()
        if scheduler_content=='No scheduler':
            scheduler_idx=0
        elif scheduler_content=='Scheduler-1':
            scheduler_idx=1
        else:
            scheduler_idx=2
        
        batch_size=int(self.batch_size_entry.get())
        checkpoint=float(self.checkpoint_entry.get())

        infer_dict=dict()
        for i in range(1,self.total_systems+1):
            if i==1:
                infer_dict['Traditional']=self.infer('Traditional',cmd_weight,scheduler_idx,checkpoint,batch_size)
            elif i==2:
                infer_dict['DeepSCS']=self.infer('DeepSCS',cmd_weight,scheduler_idx,checkpoint,batch_size)
            elif i==3:
                infer_dict['CNN']=self.infer('CNN',cmd_weight,scheduler_idx,checkpoint,batch_size)
            elif i==4:
                infer_dict['LSTM']=self.infer('LSTM',cmd_weight,scheduler_idx,checkpoint,batch_size)
            else:
                infer_dict['FRSC']=self.infer('FRSC',cmd_weight,scheduler_idx,checkpoint,batch_size)
            self.num_systems.set(i)
        
        self.frsc_infer.set(infer_dict['FRSC'])
        self.deepscs_infer.set(infer_dict['DeepSCS'])
        self.cnn_infer.set(infer_dict['CNN'])
        self.lstm_infer.set(infer_dict['LSTM'])
        self.tradi_infer.set(infer_dict['Traditional'])

    def infer(self,system,cmd_weight,scheduler_idx,checkpoint,batch_size):
        cmd_weight_frsc=torch.tensor(cmd_weight,device=self.DEVICE)
        if scheduler_idx!=0:
            scheduler=torch.load('pretrained_models/scheduler/scheduler-'+str(scheduler_idx)+'.pth')
            scheduler.to(self.DEVICE)

        if system=='Traditional':
            sc_model=Traditional()
        else:
            sc_model=torch.load(self.system_dict[system])
            sc_model.to(self.DEVICE)
        speed=0
        channel='AWGN'
        snr=10

        for batch in self.test_loader:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']

            inputs = Variable(inputs, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            inputs, targets = inputs.to(self.DEVICE), targets.to(self.DEVICE)

            temp_batch_size=inputs.size(0)
            inputs_order=[i for i in range(0,batch_size)]
            if system!='Traditional':
                if 'FRSC' in system:
                    inputs, inputs_order, cmd_entropy=scheduler(inputs,False,cmd_weight_frsc)
                    targets=targets.tolist()
                    targets_temp=[]
                    for idx in inputs_order:
                        targets_temp.append(targets[idx])
                    targets=torch.tensor(targets_temp,device=self.DEVICE)
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
                x = self.channels.AWGN(x, n_var)
            elif channel  == 'Rayleigh':
                x = self.channels.Rayleigh(x, n_var)
            elif channel  == 'Rician':
                x = self.channels.Rician(x, n_var)
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

            app_inputs=torch.split(x,split_size_or_sections=1,dim=0)
            outputs_list=[]
            start_infer=time.time()
            for app_input in app_inputs:
                outputs, finish_layers, uncertainty=self.app_model(app_input,True,speed,-1,targets)
                current_infer=time.time()
                if current_infer-start_infer>=checkpoint:
                    break
                outputs_list.append(outputs)

            command_sequence=['-' for _ in range(self.BATCH_SIZE)]
            for k,outputs in enumerate(outputs_list):
                _,top_index=outputs.topk(1)
                cmd=self.cmd_dict[top_index.view(-1).item()]
                command_sequence[inputs_order[k]]=cmd

        del sc_model

        command_infer=''
        for cmd in command_sequence[:16]:
            command_infer=command_infer+cmd+', '
        return command_infer

    def run(self):
        self.root.mainloop()

if __name__=='__main__':
    ui=TransceiverInferUI()
    ui.run()
