# *Mistral-LeChat*
  Simple and Efficient implementation of mistral models. it currently support **Mistral-7B**. compare to other architecture mistral excel at efficient cause it leverages GQA, SWA. 

  **GQA**: Group Query Attention, where in MHA-multihead attention every query in each head has thier own k, v. but in GQA we have same N query respected to head , but k, v will be shared to group of query. so it's reduce the parameters count and also kv cache size.
  
  **SWQ**: Sliding Window Attention. it's prettie cool that the way they used SWA. what would we do in standard casual attention, each token atten to every token that is previous to it's position. but what SWA does each token only atten to tokens that are within window range. now what you are thinking if we do that transformer can't understand it's context but here the twist. if we stack layer togather eventhough at each layer we do SWA but passing context to every layer each token will attention to window * layers tokens that already communicate with it's previous tokens.

# *Install*
      pip install torch transformers xformers hugginface_hub 
 
# *train*
  just clone the repo and run. 
  
      python train.py
  it will take care everything . As a default it will initiate model with 15M parameters and it will run about 5m then automatically you get samples from your trained model. if you got some issues during training or some bug , you can peek into code and figure out youself cause that's what i do 😀. 
# *What i got*
  just trained the model with default config about 5m. don't mind the words after all model trained on Eminem's lyrics 🤘
  
      look if you had one shot one opportunity 
      to seize everything you ever wanted one moment 
      And you're starting to save you
      I'm cleaning around for you
      
      You're not gonna do, but you're gonna make it
      And you're like you're just a good guy
      I'm gonna never get away from the stress
      
      [Chorus]
      I'm gonna be alone, I'm not a fuckin' me
      
      [Chorus]
      And I'm like a genius (yeah!)
