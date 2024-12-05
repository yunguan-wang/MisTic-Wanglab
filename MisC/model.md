<!--<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>
-->

# Model Details 
We for now assume that cell type information is available either through manual gating or some clustering algorithm or both. 

## Notations 
Suppose that we have $t=1,\cdots,T$ detected transcripts, $c=1,\cdots,C$ segmented cells, $g=1,\cdots,G$ genes, and $k=1,\cdots,K$ cell types. We let $||\{*\}||$ denote the cardinality of the enclosed set. By abusing the notation a little, we denote by $d(t, c)$ the distance between the $t$th transcript and the polygon of the $c$th cell.

### Directly observed quantities 
#### Cell $c$

- $v_c$: a vector describing its location (this contains its cell centroid and coordinates of its cell boundary polygon vertices). We assume that no two different polygons have non-empty intersections. 

#### Transcript $t$

- $u_t \in \mathbb{R}^2$: its spatial location 
- $r_t \in \{1, \cdots, G\}$: its gene type 

### Primary derived quantities 
#### Cell $c$

- $x_c = [x_c^1, \cdots, x_c^G]^T$: the noisy gene counts. $x_c^g$ can be computed via $||\{t: d(t, c)=0 \text{ and } r_t=g\}||$.

#### Transcript $t$

- $a_t \in \{1, \cdots, C\}$: the transcript assignment such that $d(t, a_t)=0$. This assignment is valid as long as polygons do not intersect and the set of polygons covers all the transcripts. 

### Secondary derived quantities 
#### Cell $c$

- $\theta_c \in \{1,\cdots, K\}$: cell type assignment based on $x_c$

#### Transcript $t$

- $w_t \in \mathbb{R}^i$: a vector of explanatory variables for computing reassignment probability without the cell type information.

### Non-degenerate variables 

- $\delta_t \in \{0,1\}$: an indicator of misassignment. 0 means correct assignment while 1 means mistaken assignment. 

### Tertiary  derived quantities 


- $b_t \in \{1, \cdots, C\}$: the nearest neighbor cell of the $t$th transcript of a different cell type than cell $a_t$
- $z_t = [m_t^T, e_t^T]^T \in \mathbb{R}^{l}$: a vector of explanatory variables for computing reassignment probability where $m_t \in \mathbb{R}^{l_1}$ and $e_t \in \mathbb{R}^{l_2}$ are features based on distances and expressions, respectively and $l=l_1+l_2$. $z_t$ is computed based on $\{\theta_c, x_c, v_c\}$ and $\{u_t, w_t\}$.

### Degenerate variables 

#### Cell $c$

- $y_c = [y_c^1, \cdots, y_c^G]^T$: the true gene counts. 
We note that given $\{a_t, b_t, r_t, \delta_t\}$, we have for a given gene $g$ and a given cell $c$
$$y_c^g=x_c^g-||\{t: \delta_t=1 \text{ and } a_t=c \text{ and } r_t=g\}|| + ||\{t:\delta_t=1 \text{ and } b_t=c \text{ and } r_t=g\}||$$

## Features 
For features based on expressions, we will use Deseq2 to compute the log2 fold change: log2FC and the adjusted p-values: padj. We will use $\text{log2FC}*(-\log_{10}(\text{padj}))$. To prevent small p-values from yielding extreme values, we compute the percentage ranks of the products in an ascending order $\{\pi_t\}$ and transform by log odds $e_t = \log(\dfrac{\pi_t}{1-\pi_t})$.

For features based on distances, we will use the percentage ranks of the distances from transcripts to the boundary of the cells they are assigned to (without cell type information) or to the boundary of the cells of their nearest neighbors (with cell type information) and transform them by log odds. Unlike the expression features, the ranks are computed within a cell. 

## Model 
Although we can model the observed gene counts, to align with our current pipeline, we choose to model the cell types. Given the size of a typical SRT data, we opt for VI. (For other alternatives, see google doc for their sketches).

We choose to model the log conditional probability: 
$$\log p(\{\theta_c\}| \{v_c\}, \{r_t, u_t\})$$

$$
\log p(\{\theta_c\}|\{v_c\}, \{r_t, u_t\}) = \int p(.) \log \dfrac{p(\{\theta_c\}, \{\delta_t\}|\{v_c\}, \{r_t, u_t\})}{p(\{\delta_t\}|\{v_c, \theta_c\}, \{r_t, u_t\})}d\{\delta_t\},
$$
where 
$$p(.) = p(\{\delta_t\}|\{v_c, \theta_c\}, \{r_t, u_t\})$$

### Numerator within the logarithm 

$$p_\phi(\{\theta_c\}|\{v_c\}, \{\delta_t, r_t, u_t\})p(\{\delta_t\}|\{v_c\}, \{r_t, u_t\})$$

In the first term, we note that we have 
$$p_\phi(\{\theta_c\}|\{v_c\}, \{\delta_t, r_t, u_t\})=p_\phi(\{\theta_c\}|\{x_c,v_c\}, \{\delta_t, a_t, r_t, u_t\})$$

For a specific configuration $\Theta$ of $\{\theta_c\}$, we would have arrive at a configuration of $\{b_t\}$: $b(\Theta)$. Therefore, strictly speaking, the conditional distribution is a mixture of category distributions. However, since we have access to the cell type information, we only need to worry about one of the many possible configurations $\{b_t\}$. With $\{b_t\}$, we can compute $\{y_c\}$. And we suppose that the assignment is generated from a logistic regression governed by $\phi$.

Since the likelihood contains no spatial information, nowhere in the model does it inform the model that for two transcripts of the same type, reassigning the one that's closer to the boundary (self or neighbor) is favored. 

Therefore, we use the second term $p(\{\delta_t\}|\{v_c\}, \{r_t, u_t\})$ to inject this knowledge. Using $\{v_c\}, \{r_t, u_t\}$ we can compute $\{w_t\}$. Hence, a priori, the probability of reassigning a transcript is an other logistic regression with fixed parameters. 

To assign parameters, we need to specify without knowing the deseq2 results, how "comfortable" are we to reassign a transcript that's ranked 50% and how "comfortable" are we to reassign a transcript that's ranked 5%. We are using 0.01 and 0.8. 


### Denominator within the logarithm 

$$p(\{\delta_t\}|\{v_c, \theta_c\}, \{r_t, u_t\})$$

Although we do not know the true posterior, we can approximate it via 
$$q_\beta(\{\delta_t\}|\{v_c, \theta_c\}, \{r_t, u_t\})$$
As we assume that $\{z_t\}$ is determined by the information that we are conditioning on, we can simply re-write it as 
$$q_\beta(\{\delta_t\}|\{z_t\})$$

### ELBO 
$$
\log p(\{\theta_c\}|\{v_c\}, \{r_t, u_t\}) \geq \int q_\beta(.) \log \dfrac{p_\phi(\{\theta_c\}|\{v_c\}, \{\delta_t, r_t, u_t\})p(\{\delta_t\}|\{v_c\}, \{r_t, u_t\})}{q_\beta(\{\delta_t\}|\{v_c, \theta_c\}, \{r_t, u_t\})}d\{\delta_t\}
$$

We use the Gumbel softmax trick to sample Bernoulli RVs and we can learn all the parameters via gradient decent. 

The loss is then $\text{cross entropy loss}(\theta_c, \text{logits}) + \text{KL Divergence}$



## Sampling 
We will patchify the data. Within each patch, we make sure that there are enough cells (but not too many). 


## Quick summary
The framework allows simultaneous learning of all the parameters without grid search. However, there are several issues 
### Issue 1
Features. We need to contemplate on the features we construct. 
### Issue 2
Based on the ELBO alone, nothing is telling the model that reassigning a transcript that's far away from the boundary is bad. Three possible solutions: 1. construct better features; 2. add penalty on distances; 3. make the coefficients are positive. (I have observed negative or close to 0 coefficient corresponding to distance feature).   
### Issue 3
Cell typing. We are assuming cell typing is fixed. Do we need to relax this assumption? If so, we will need to consider modeling the observed gene counts $x_c$. 
### Issue 4
Patch generation. Patch size. 
### Issue 5 
Previously we are using entropy. We can still add entropy into the loss. Or we can use entropy as a validation. A potential drawback of entropy is that its minimum is attained at various points. When the classifier is super certain that the cell is or is not of a type, the entropy will decrease in both cases. 

## Issue 5
We kind of justify why we do not need to re-do cell typing if we model the observed gene counts.  

Now, we only observe $a_t, u_t, r_t, v_c, x_c$. Note that we do not have $b_t$ since it's determined by $\theta_c$.

If we model 
$$p(\{x_c\}|\{v_c\}, \{a_t, u_t, r_t\})$$
following a similar step, we have on the numerator in the logarithm 
$$p(\{x_c\}, \{\delta_t\}, \{\theta_c\}, \{y_c\}|\{v_c\}, \{a_t, u_t, r_t\})$$
we decompose it into the product of 
$$p(\{x_c\}|\{y_c\}, \{\theta_c\}, \{\delta_t\},\{v_c\}, \{a_t, u_t, r_t\}),$$
which is deterministic since given $\theta_c$, we can now figure out $b_t$, and 
$$p(\{y_c\}|\{\theta_c\}, \{\delta_t\},\{v_c\}, \{a_t, u_t, r_t\})=p(\{y_c\}|\{\theta_c\}),$$
which could be a fancy zero-inflated negative binomial model or whatever, and 
$$p(\{\theta_c\})p(\{\delta_t\}),$$
assuming that we do not wish to model impact of spatial information on cell typing. 

On the denominator in the logarithm, we have 
$$q(\{\delta_t\}, \{\theta_c\}, \{y_c\}|\{x_c\}, \{v_c\}, \{a_t, u_t, r_t\})$$
we decompose it into the product of 
$$q(\{y_c\}|\{x_c\}, \{v_c\}, \{\delta_t\}, \{\theta_c\}, \{a_t, u_t, r_t\}),$$
which is deterministic since given $\theta_c$, we can now figure out $b_t$, and 
$$q(\{\delta_t\}|\{x_c\}, \{v_c\}, \{\theta_c\}, \{a_t, u_t, r_t\}),$$
which would be the same as the one we have in the current model as we can know compute $z_t$, and 
$$q(\{\theta_c\}|\{x_c\}, \{v_c\}, \{a_t, u_t, r_t\})$$
Of course we model this however we want. But notice that the condition information only contains the observed gene counts. As nothing is updated during the learning, we do not need to re-do cell typing under this setting. 

A con of modeling gene counts versus cell type is that it involves more parameters (2-3 times more). 

Now, back to our original setting where we model the cell type instead of the gene counts. Maybe we can justify as follows: 

$$p(\{\theta_c\}|\{x_c,v_c\}, \{a_t, b_t, u_t, r_t\}) = \dfrac{p(\{x_c\}|\{\theta_c, v_c\}, \{a_t, b_t, u_t, r_t\})p(\{\theta_c\}| \{v_c\}, \{a_t, b_t, u_t, r_t\})}{p(\{x_c\}|\{v_c\}, \{a_t, b_t, u_t, r_t\})}$$
If we choose not to incorporate spatial information into cell typing, we can treat the second term on the numerator as fixed. The denominator is what we just discussed if we remove $b_t$. Therefore, once the initial cell typing is done, we do not need more iterations. 


