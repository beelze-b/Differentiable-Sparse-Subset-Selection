# Feb 18

http://www.jobs.cam.ac.uk/job/27517/

### Notes For Xie and Ermon Stanford Paper

Questions

* How does the extension get around requiring a combinatorial number of categories?
* The goal is to do gumbel max such that you get gradients and relaxations
* What are the advantages compared to Learning to Explaining



Does not work when $0<t<1$



## Notes from Meeting	

1. Try to binarize the selection layer
2. Try to do it form the squeezefit point of view

try $$W = Y Y^T$$

$Y \in \mathbb{R}_{n \times k}$

$$||Y_i|| < 1$$ where each row is less than 1

$$\sum_i \lambda ( || Y_i||_2^2 - \frac{k}{n})^2$$

Not exactly Squeezefit since SqueezeFit can give you W with rank greater than K.

Not guaranteed to converge because not convex



Still have the latent representation stay the same

Test sparsity by looking at the eigenvalues of $W$

Possible to get a lower value of k. 

Selecting basis to project basis. 

Look at rank $W$

Interesting in the MNIST dataset. Maybe even k = 10 can work

Try with a higher k.

3. **Zeisel dataset**

Mice brain cells can compute how many times a gene is expressed in a cell

It is a sparse dataset and try to get the most representative genes.

Can we hope to get something lower to 500 genes?

8000 cells, 20,000 genes and want to use 500 genes

Gave me a post-processing matrix.

These cells belong to different classes. There is a class file as well. Tried SqueezeFit on it and it worked well.

Removing it from SDP

SqueezeFit method is a linear representation, so try a non-linear. 



Much stronger selection because it can reconstruct and select non-linearly.

Can the latent space go to labels? Can you add a constraint such that latent space goes to labels?



### Goals for Next Week

1. Try Binarized Selection layer for MNIST
2. Try the SqueezeFit like penalization for MNIST with $W = Y Y^T $ as above
3. Try W diagonal on Zeisel data. Find markers for reconstruction. Can the labels find a latent representation?
4. Look at code by Xie and Ermon



1 and 2 MNIST

3 are Zeisel



Generation

# Feb 24 --- Meeting with Bianca

Most people seem to not have looked at $L_1$ penalty because they saw it as too basic.

Papers to read:

* Xie Ermon - Recontinuous --- read again and implement code
* Concrete Distribution
* Learn to Explain
* Sparse Latent Representation from Kyungyun Cho group
* Learning with Differentiable Perturbed Optimizers
* DeepPINK: reproducible feature selection in deep neural networks
  * reproducible feature selection
  * want to show that our paper would lead to better results
  * Kinda like a permutation test
* Joint Autoencoders from Uhler
* Explainable Models from 

Want to select a lot less Genes. We are selecting 220 features right now.

Goals:

1. Match in Latent Space
2. Also want to maintain separability in the classes
3. Look at the embedding space
4. Comparison to a linear program like SqueezeFit
5. Is there a way to train the autoencoders together?
6. What would be good synthetic data?

What would the objective look like?

**What is the Gumbel trick?** Pretty much like the reparam trick from VAEs. IF satisfied on the simplex is supported at the vertices, you have a discrete distribution.

Looking at subsets of k from n, reducing the number of free parameters to make it tractable

What is our advantage? You have a distribution on a simplex too, but a different distrubtion and also encourage sparsity in a different way. Do a Benchmark paper. Xie and Ermon has the advantage of selecting exactly k things, but how do they back propagate the gradients through w?

$\texttt{Reconstruction Loss} + \texttt{Points in Same class are closer than points in further class}$

Use L1 and and then try Gumbel-max paper. What does it mean from a practical point of view? What does this mean for explainable models? Wainwright and Beanu (?) paper might provide a better context than the genetic data.

What is the convergences? Trying different initialization? 

How to take derivatives with respect to a linear program, if we wanted to use SqueezeFit directly?

Sparse latent representation? -Joan Bruna maybe we can fit it to our case?

Training Coupled Autoencdoers

Joint-Coupled Autoencoders with KL Divergence by Uhler, maybe non-llinear compressed sensing

**My Next Tasks**:

1. Push Notebooks to github
2. Jointly Trained Model autoencoder
3. How does Loss and Sparsity change as a function of the regularization parameter? Diagonal and matching latent space
4. Incorporate the Gumbel softmax trick into VAE setting.

Try on the Zeisel data. MNIST data requires thinking about th ecentering and images.

### 3/10

Trying $|| ||w||_1 - 50||_2^2$

Xie and Ermon make sense in a variational framework. Where does L1 norm make sense?

Why does L1 norm not work? You cannot take good derivatives with it. And the objectives is non linear.

Learning Sparse Neural Networks Through $L_0$ Regularization --- https://arxiv.org/pdf/1712.01312.pdf

Maybe this can work? with l_1

See if $L_1$ penalty have been used successfully for neural network training

Try the Xie/Ermon technique

SqueezeFit approach tries a different approach

Tasks

1. maybe try different lambdas or increase lambda over time to see how it works
2. Try the joint training approach with gumbel and the l1 norm
3. Switch back to all data points have same subset selection
   1. better for the genomic approach

Read:

L2x http://proceedings.mlr.press/v80/chen18j/chen18j.pdf



#### Concrete https://arxiv.org/pdf/1611.00712.pdf

	* What is it for? A: reparamatrization trick for discrete variables that is unbiased and low variance

* https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html includes a nice summary
* essentially just applying softmax
* *For small temperatures the samples are close to one-hot but the  variance of the gradients is large. For large temperatures, samples are  smooth but the variance of the gradients is small.*

Relax-k https://arxiv.org/pdf/1901.10517.pdf

Through l_0 https://arxiv.org/pdf/1712.01312.pdf 

* using concrete distribution with hard sigmoid to apply hard gates to weights
* is more about L0 reagularization for whole network, not just for features
* l1 might still be important because it takes into account scale (is a feature more important than the others)

What is the state of neural network pruning https://arxiv.org/pdf/2003.03033.pdf

* seems more like a review

Deep Pink https://arxiv.org/pdf/1809.01185.pdf

* feature selection by controlling input error

Learning Controllable Fair Representations https://arxiv.org/pdf/1812.04218.pdf

* tries to unify many methods by making them controllable
* actually goes into the Learning Fair Representations from Responsible Data Science class

Generative Modeling by Estimating Gradients of the Data Distribution https://arxiv.org/pdf/1907.05600.pdf

* seems for generative procedures, not feature selection directly
* maybe can be co opted

Invariant Risk Minimization --- https://arxiv.org/pdf/1907.02893.pdf



Lasso, fractional norm and structured sparse estimation using Hadamard product parametrization (Dunson's Paper): https://arxiv.org/pdf/1611.00040.pdf



Supplement:

on Bayesian priors: https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html

Dropout as as structured prior: https://arxiv.org/pdf/1810.04045.pdf

### Questions

1. How does this relate to attention modules?



### Meeting Professor Villar March 24

What L0 method do we have?

Plot reconstruction error vs sparsity

1. Try Jointly trained VAE with gumbel
2. pre traind VAE with Gumbel
3. Try normalizing L1 method (normalize the weights before)
4. read the paper

Line plot of sparse and reconstructed values side by side



## Meeting April 30th

1. The model i am using that it is a mix of Soledad's and Bianca's might be too complicated
   1. it is a non-linear factor analysis model
2. Use one from Soledad and Bianca would jointly make
3. How does Gumbel softmax work in simpler settings? Works for MNIST
4. Check that Gumbel trick is actually working on MNIST and synthetic data!!!!!
   1. ~~use pre-defined model for MNIST part too~~
   2. check for k values in gumbel trick on synthetic data
   3. Try hyperapram tuning
   4. Try simpler synthetic data
5. Include L1 trick for the synthetic data
6. Read paper



What are the three methods to try out?

1. Gumbel trick
2. L1
3. Gradients
   1. doesn't matter that it is instance-wise for genetic samples



Notes

1. Try Gumbel trick per batch
2. L1 trick on synthetic data in noisy and not noisy situations
3. Move stuff to overleaf document
4. Try out memory bank





### Call with Bianca's colleagues @ May 11, 2020

Lots of literatre on feature selection in terms of clustering

* Bayesian variants
* our focus is not cluster/classification but compression
* **How does shrinkage prior relate to the Gumbel softmax? Sampling what vs sampling dirichlet**
  * while making it all differentiable
* **double exponentials can give you an L1 penalty**
  * the conditional distribution of a hierarchical model changes a lot the marginalized distribution
* **penalty on the graph structure** by professor we talked to
  * relates to different L1 penalties (for tree structured data)
  * Hutchinson's papers
* we are sparsifying the data, not the network exactly
* Emily Fox: VAE feature selection
  * similar to Bianca's non-linear CCA paper
  * does not solve interpretability because does not constrain the input you want
* problem of identifbiality
  * different results based on different trials are inherently not interpretable
* smart places or ways to do L1
  * gradient descent runs into issue
  * **Peter Hoff's trick of combining L2 penalties to induce L1 penalties**
    * always working with L2 penalties when the gradient is concerned
* scalable nonlinear factor analysis method would be really cool
* so far this isn't a good paper because we do not show anything reproducible to do
  * are there a class of shrinkage priors that give us more identifiability
  * how to write down the neural network architecture?
  * How to trust any of it? Maybe simplify it?
  * robust to pertubations of the data
  * he does not care too much about the gradients
  * **can we get consistent feature selection across multiple runs**
* **Converting a prior into a penalty canonical paper**





### Call on May 25, 2020 with Federico Ferrari on Non Linear Factor Analysis

* not directly related to problem, but has similarities

### Call with Soledad and Bianca on how to make my paper different from Concrete-VAE --- June 29th

**Venues**:

* AISTATS
* **Nature Methods (great for everyone)**
* ICLR

**Datasets**:

* simulated datasets
  * kind of the way i am doing
    * ![image-20200629150649475](/home/nabeel/.config/Typora/typora-user-images/image-20200629150649475.png)
  * when f and g are linear, how does it compare to scGeneFit?
* Concrete VAE Datasets --- they have gene datasets
* Zeisel Datasets
* Other development cell datasets that Bianca would get

**Methods to Compare**:

* Ours

* Concrete VAE

  * how do they select distinct top k? doesn't seem like they do

* L1 Loss

* add in GP-LVM, if our method is worse

  

**Advantages**:

* very good benchmark paper that is very transparent that is very good for community

**Tasks**:

* compare to gumbel in both (instance and overall---concrete VAE)
* maybe include scGeneFit
* **Do I need to re-code concrete AE?**

**If we cannot make anything better** (just as good or slightly worse), can still generate benchmarks on a bunch of biological papers. What are the interesting things they want to do:

* single cell trajectory	
  * trajectories in T-SNE can be biologoically interpretable along some axis (x_1 is time)
* GP-LVM

**Metrics to compare these methods**:

* Graph out the logits of the average of experiments to make the color graphs smoother
* How does it deal with compression in the latent space?
* How do they do when using fewer markers?
  * k vs RMSE
  * do this on synthetic vs non-synthetic datasets
* How do the dynamics of training change when you from variational autoencoder to a varaitonal autoencoder with gumbel/concrete?
  * what is the effect of this regularization on this approach?
* Number of batches or epochs for convergence?
* Datasets
  * Synthetic
  * Zeisel
  * L-1000 From Concrete VAE



Notes from Development Session on August 9, 2020

* it seems that nothing really beats using gradients of vanilla VAE to select features in the noisy data scenario
  * does that mean might as well use gradients? does this work for "real" datasets?
  * gradient difference is sharper with my technique (See notebook 11 with logits lambda = 0)
  * logit loss makes the gradient trick not work but improves probability of making subset trick work (lambda = 0.001)
  
* Notebooks 9 through 11 are like scratch work notebooks.

* This method is better.

  * Try with the same architecture

  * Why doesn't it select distinct? IT DOESN"T

    ![image-20200810043703213](/home/nabeel/.config/Typora/typora-user-images/image-20200810043703213.png)

* what made VAE gumbel give the better results for simplified concrete AE and batch gumbel and vanilla VAE gumbel was anneal temperature better, have better initial and end states for the temperature, and do more epochs

**Questions to ponder**

* Does it train to the same accuracy after enacting burnin in mode compared to vanilla?
* The reconstruction loss is between the logit encoding and the hidden state. Should it between the logit encoding and the initial logit?
  * This can be very unstable.
* Everything is sigmoid decoder and cross entropy loss. Is this right?
* ~*Should I re-code concrete AE into pytorch?~* Done
* How much does encoder and decoder complexity matter?
* Is it all about selecting the non-noisy features? What if some of the non-noisy features are really correlated, so it can pick a noisy feature to get higher accuracy? If it chooses to pick more noisy features, it must have higher accuracy!
  * Even then, this is not necessarily great for another task
* Are we subset sampling or concrete sampling?
* Should we list top logits for concrete vae to be all distinct or include the non-distinct ones as well even though it is diffrent from how i generate the topk for mine
  * Even then, when I do it the distinct way, it isn't too different 
  * but it is useful to look at the distinct ones anyways and then do non-distinct ones when counting how distinct
  * Plus the way of looking at top logits in my methods that doesn't actually simulate doing the k steps is pretty close anyways, if the updates are pretty much putting all the way on the top logit
    * this is because the order of the top logits is pretty close to the simulated method as long as all the logits are spread out or there is a low temperature (and they are all of the same scale
* You might wonder if when looking at our methods, if they look bad when we allow duplicates or when we allow distinct. When we allow duplicates, the duplicates happen in the real data section (except for vanilla VAE gumbel). When we don't allow duplicates, usually the real features are filled out. When we allow duplicates int he concrete stuff, they sample in the real data, but deduplication pushes them into the synthetic data.
  * Their stuff is as good as vanilla VAE gumbel.
* How does distinct capture rate depend on what percent of the features are noisy (50% so far)

**What do the notebooks show:**

* **Notebook 8** shows that gradients are usually better than indices for both VAE Gumbel and VAE. It works just as well in VAEs, but VAE gumbel helps out. Gradients are pretty clear in picking one set, but can pick one set over the other all the time. Normal VAE gets it right more often though. 
  
  * This shows an early version of why [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/a2394d961bb7206521776c1f0fe80597b7d57ea1/notebooks/8_SyntheticData.ipynb) is
  * Another [example](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/5ec6817256b1fb46609d6ef8230bf3452cf596f6/notebooks/8_SyntheticData.ipynb)
  * Note the old links were when the encoder and decoder were more complicated.
    With a more complicated encoder and decoder, maybe the gradients have a higher chance of getting right in both VAE and Gumbel cases. However the second old link has goes against this idea. **Might be because I didn't run for enough epochs and that would suggst Gumbel is just a little more complicated.** Runs of notebook 11 with low number of epochs show [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/8a17e1c62caa31b3a9b7c7b1fcd7256648151e1b/notebooks/11_StableGumbelWithState.ipynb) here (the commit message shows the configuration of the three VAE Gumbel approaches)
  
* Notebook 9 shows that gradients do easier than indices for VAE Gumbel when testing a subset of indices. 

* **Notebook 10** shows that batching gumbel improves the performance in both truncating the loss and with noisy features, but it is not always competitive with the gradient method. gradient method suffers a little. Batching keeps the orientation right, compared to 8.

* Notebook 11 compares VAE Gumbel with my VAE Concrete with one set of weights. My VAE Concrete is not the best when it comes to selecting features, but it seems the most stable. Went through various evolutions of simple concrete vs complex concrete. Include all the methods at the end

  * VAE Gumbel and BatchingGumbel both show a gradient exploration vs gumbel selection goodness trade off. 

    [This run of notebook 11](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/7490b7f836f66365b87c3657226868e4cb9ec606/notebooks/11_StableGumbelWithState.ipynb)	compare to [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/5ec6817256b1fb46609d6ef8230bf3452cf596f6/notebooks/8_SyntheticData.ipynb) run of notebook 8 for Vae Gumbel shows this behavior

    ```
    To get good gumbel selection, you might need more gradient exploration during burn in. You could of course get lucky and your gradients find the right spot and your gumbel selection then finds the right spot. Or your gradients could get somewhere else.
    ```

    [This](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/b1351a7715d1ac7fa3718bcd31a29f22d187dda4/notebooks/11_StableGumbelWithState.ipynb) shows the previous commit where we got lucky for VAE Gumbel: 

  * did experiments with both our simple and complex concrete VAE

    * [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/7490b7f836f66365b87c3657226868e4cb9ec606/notebooks/11_StableGumbelWithState.ipynb) is a good version of the simple one 
      * gradients blew up for vanilla VAE. Is this because of numerical stability? is this because of convergence?
    * [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/c6b17bf76f4ddc77950d3317262a148c8e83cda5/notebooks/11_StableGumbelWithState.ipynb) shows that the complex one with alpha = 0.9 and logits_lambda = 0.001 is almost random, but still better than random
      * but [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/c0e5f92f1113cfe04f7ed526dc5a0f48d4e1afc4/notebooks/11_StableGumbelWithState.ipynb) shows the same with alpha is 0.99 and no logits loss so i think this method is fruitless
    
  * actually complex our concrete vae might work after i fixed a detachment bug and gumbel keys in place (I assumed pre_enc and logit_enc could not change in my moving average work)
  
    * with alpha = 0.99 and logits lambda = 0, logits loss still went down over time
      * 7000 logits loss in first few epochs, to 2000 logits loss around 150 epochs, rose to 30,000 to epoch 300,
    * [this](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/21ff9ec8ed44d92ee050922d170ae49c78521036/notebooks/11_StableGumbelWithState.ipynb) shows that it is actually the best. with alpha = 0.9 and lambda  = 0, has best selection, most pure, best logits, and best at selecting distinct k (things close to 1 or 0)
  
* Notebook 12 runs our version of their concrete AE and seems to have similar performance to their version of it

* Notebook 13 builds k models that can be exported as .py for a cluster

* Notebook 14 is about visualization

* Trial 1 of RunningState and Concrete VAE were run locally

  * Can use the other trials to analyze how they did
  * Did this because Logit Enc didn't save for RunningState and the top logits/subsets for ConcreteVAE weren't as good as I expected
  * This might have been premature because running through those other trials Concrete VAE automatically shows thaty they were alright
  * **FUCK I JUST DIDN"T LOAD THE MODEL** THERE WAS NO ERROR
  
* Notebook 15 - Zeisel visualization

  * global gate did the best and concrete vae and running state had trade offs with each other
  
* Notebook 16 - More zeisel visuazation

  * seems like running state might have better indices but its reconstruction is not as good.
    * trying to increase the alpha to .9 and add another layer to weight creator. might be there that the friction there is too high. **What if i could add attention instead to weight the instances better.**
  * think it is running off utils from gaussian decoder branch
  
* Notebook 17 - Paul Dataset

  * Using utils off gaussian decoder branch
  * 

**Notes on Each Method:**

Vanilla Gradients:

* gradients can be unreliable in vanilla 

L1 Weights:

* can't control k

Vanilla Gumbel:

* gradients and gumbel logits can select the exact opposite!
* Did not require annealing in one experiment!
* **slightly ahead of One Set of Logits even without annealing!**
  * [Link](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/b1351a7715d1ac7fa3718bcd31a29f22d187dda4/notebooks/11_StableGumbelWithState.ipynb) **THIS COMMIT HAS GREAT NOTES TOO**
* gradients are less stable here and can blow up [maybe](https://github.com/beelze-b/Differentiable-Sparse-Subset-Selection/blob/7490b7f836f66365b87c3657226868e4cb9ec606/notebooks/11_StableGumbelWithState.ipynb)?

Batching Gumbel (haven't done with more careful annealing and more epochs yet):

* average of the logits is usually consistent
* dependant on batch size
* logits can be really expressive because logits come out of a network (more of a pro against *one set of weights* and *concrete ae* models)

Moving Average of Logits Passed to Gumbel  (haven't done with more careful annealing and more epochs yet):

* lots of things to tune
* can weight certain instances more
* not necessarily better than concrete AE and was sometimes random
* more useful than vanilla gumbel since points in right direction, but worse than batching gumbel in terms of consistency and less useful in degree of difference in logits
* lets you insert prior
* **forgot to save logit_enc so don't have data yet**

One Set of Logits for Gumbel (much like Concrete AE):

* seems to be a little slower in getting same accuracy as Vanilla Gumbel
*  

Concrete AE:

* treats the logits as independent ??
* not necessarily distinct
* Chooses features differently, so having things that aren't close to zero or 1 is actually worse than our case since they do dot products (and could mix features)
  * Our other methods just scale features and don't "mix"
  * However our method might mess up the smoothness of the landscape of the gradients because we might scale up features
  * essentially a difference a relaxed k hot vector and a matrix multiplication to imitiate selection by k relaxed 1 hot vectors



Notes on Batching With a state but still a logit for each instance:

​	Pros: Can weight each instance. Maybe can use for initialization

​	Cons: Hard to tune and less intuitive



#### Notes for Meeting on September 23 10 am

* removed burn in period
* use momentum method that is sligthtly modified from batch nrom in terms of clipping gradients
* gradient methods can pick the wrong features in terminated early
* having something create weights lets the weights be more expressive if needed
* my method works because I fixed a bug in the update of the gradients (I forgot to detach logit_enc by making it  parameter too early)
* Need to run for enough epochs for learning to settle down

![Notes of that meeting](/home/nabeel/Documents/NYU/SecondYear/SecondSemester/Differentiable-Sparse-Subset-Selection/notes/imgs/PXL_20200923_153502054.jpg)

#### Meeting October 14 --- meet again in two weeks

Classification Rate experiment

* accuracy of reconstruction

  * inner product between original signal and new signal with the right normalization
  * use Code from Optimal Marker Selection by Soledad

* **variation of error vs k**

  * FORGOT THIS DUMMYYYYYYYYY

* https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03641-z

  * use the metrics from this paper to evaluate the marker gene set.
  * Try it on paul dataset

* Compare to linear method

  * linear method is supervised (SqueezeFit with the basis being the canonical basis)

* **Bianca reported that it wasn't learning, might as well fix the sparsity issue**

  * Trying not minmax scaling but just normalizing while not destroying sparsity
  * adding batch norm
  * switched to a gaussian decoder
  * reduced KL divergence regularization
  * changed indexing of batch printing
  * did xavier initialization
  * Switched to LeakyReLU for last layer
  * drastically reduced learning rate since already running for a bunch of epochs
  * *could be overfitting, should implement early stopping*
  * *Switch to colab and try a bigger network*
    * trying on colab
    * actually working better with smaller network because the average is closer
    * Running State was still more successful when giving more layers
      * [lower capacity](https://colab.research.google.com/drive/13jhAJ6-cItaprJoItGHx6n39pXB9Wvwa#scrollTo=sNItQz9BFxNe&uniqifier=1)
      * [higher capacity](https://colab.research.google.com/drive/1-2vNjl7GLoLzQMcS9DAGUrszNXRC0G2v#scrollTo=kVB7Y29DFxMW)
        * concrete has worst training loss too
        * best reconstruction overall from running state model
  * bias on or off? -- trying to not use
  * summarize models funciton
  * try on colab and bringing back:
    * bigger ntwork on colab
    * bring back bias and size settings

* **Nature Genetics** --- super high profile

  * requires a couple more analysis and datasets

  * Does the subset selection compare to a linear method?

  * spatial transcriptomic data 

    * gene expression and spatial location of cell

    * markers for cell to cell interaction

      * X is a n by d matrix 
        n is number of cells 
        and d is the number of genes 
        Z n by 2 
        position of cells 
        X’ = n^2 by 2d 
        [x_i x_j] 
        x_i in R^d and x_j is in R^d 
        X’ is x_i and x_j are within epsilon radius with Z 

      **Can we reconstruct both and get the markers?**

* computer science venue --- what is the story?

  * is the method different enough from xie ermon or concrete vae from a theory perspective?
  * not a compellling argument
  * ICLM (February), ICLR or NeuroIPS (May)
  * hard to force the theory
  * Caroline Uhler --- training dynamics
  * Still hope to get a theoretical result along the way and still submit to NeuroIPS or something along the way

* leanings

  * Nature Genetics track

* Can we extend our method to use classification loss to make it supervised?

* Three Context

  * Clustering --- compare with Optimal Marker Selection before
  * Pseudotime --- modify loss to use pseudo time
  * Spatial transcriptomic 

* FOR NEXT TWO WEEKS COMPARE WITH THE Optimal Marker Selection PAPER to see if the non-linear method is actually better

* Start developing it as a suite that you can install

Meeting December 17

* seems like they can handle it from here

* just add metrics stuff to zeisel as well in the style of paul

* don't forget to add the cosine angle stuff to both

  * Zeisel data [link](https://colab.research.google.com/drive/1u9KQfwdIwvqbVjgzwwzy0j4SgloomlDc)

    * Global Gate is best on all metrics.
    * Running State is a close contender. Unlike for Paul, we also beat on cosine angle.
    * [More complete version](https://colab.research.google.com/drive/1wd9PU7ni3Mq_165nX5EWfBcnjXh9jWvh)
* Paul Data: 
    * So ConcreteVAE has best cosine angle but worse accuracy and graphs?? [link here](https://colab.research.google.com/drive/1yVN8K6tBxGUqKZsmT7uc_B95K2RYJ9vC)
    ConcreteVAE can actually boost the reconstruction whereas we can only match the BAC/AC and not boost it with RunningState
  * Here is [one](https://colab.research.google.com/drive/13TKdqYw2YCGsFBihAPGvudLHeBLo0-Lw) of the same that looks like something for concrete vae but running state does slightly worse
    * [A version where I fix the logits being used for running State](https://colab.research.google.com/drive/1jqnOCeUfpL9mYC3hSjiGO-RmIcuIGPEz)
      * We are comparable to 84Markers from RankCorr SUCCESS!!!
      * **Best version including visualizations**
    * [Paul data when using LightGBM as classifier](https://colab.research.google.com/drive/1TqJc4FvMPi4_UK8YNKSGT5KMgxK6nfOq)
      * ranking don't change
        

Work on Jan 18

* [Experiments with different classifiers](https://colab.research.google.com/drive/1C2zBiSTWx3n_PhoPEXKTkrUYVo86_1U2)

  * only when usiing a super optimized GBM could i reach the high 73% accuracy on the test set
  * kinda doubtful of their results
  * Going to use their own markers on our test data (and their markers should overfit nonetheless)
  * Should I normalize the data differently (log normalize?)
    * switch to log normalization gave comparable results on the RFC so probably not worthwhile route

* need to have 16 and 17 notebooks with the final classifiers

  


### Log Events

#### December 1st, 2020

* multiple k fo rtraining zeisel
  * 
* top logits for visualization and accuracy in paul
  * 





December 10th, 2020

* thinking about how to attentionalize the recurrent nature of the xie ermon paper
* if we want k features, have k matrices and just make sure that the first column of M_1 is orothogonal to the first two columns of M_2, and that the first 3 columns of M_3 and orthogonal with the previous main columns
  * mulitply the hidden representation of a feature by all these matrices and then apply concrete. hopefully if k = 3, three directions could factorize well.
  * will be less distinct than runnigstate but more distinct than concrete vae, can be parallelized but less memory efficient
    * can maybe do one matrix and just make constraint that it must be orthogonal to itself
    * can you start out with a random orthogonal matrix?
