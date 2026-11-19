# SS-MinimalSurfaces. Modelo de Isin't

Para (V,E) grafo asociado a la cuadricula de $[0,N]\times[0,N]$. Dada $a:V\to [0,1]$. Veremos el modelo de Isin't, $\sigma:V \to \mathbb{Z}$

$$\mathbb{P}(\sigma)\propto exp(\dfrac{-\beta}{2}\sum_{i\sim j}[(\sigma(j)-a(j))-(\sigma(i)-a(i))]^2)$$

Consideramos condición de borde igual a cero, es decir trabajamos sobre E={ $\sigma:V \to \mathbb{Z} \text{ t.q } \sigma|_{\partial V}=0$ }.

## Glauber


Si tenemos un $\sigma$ y olvidamos lo que hace en $j\in V$. Su ley condicional se dada por 

$$\mathbb{P}(\sigma(j)=n|\sigma|_{V-j})\propto exp(-\dfrac{[n-\left( a(j)+\dfrac{\sum_{i:i\sim j}\sigma(i)-a(i)}{4}\right)]^2}{2(\dfrac{1}{2\sqrt{\beta}})^2})$$

Entonces, tiene la ley de una gausiana a valores en $\mathbb{Z}$. Ver lab enviado por wsp para encontrar su implementacion.

## MH
Escojemos un vertice $j$ uniformemente, y simulamos $v \sim \text{Unif} ( \lbrace -1,1 \rbrace )$. Definimos $\sigma^* = \sigma$ en $\lbrace j \rbrace ^c$ y  $\sigma^* (j)=\sigma(j) + v$. Aceptamos $\sigma^*$ usando aceptación-rechazo, ,el criterio queda 

$$\dfrac{\mathbb{P}(\sigma^*)}{\mathbb{P}(\sigma)}=\exp({-\beta v (2v+4\sigma(j)+4a(j)-s(\sigma,a,j))}),\text{ donde } s(\sigma,a,j)=\sum_{i\sim j}\sigma(i)+a(i)$$


Para $\beta<2.5$ usamos Glauber, y para $\beta \geq 2.5$ la dinámica de Glauberno mezcla bien

## Swendsen Wang 1986

Para $\beta >2.5$. Para $a$ dado, se simulan a la vez copias del sistema en una escalera geométrica de temperaturas $\beta_1<\dots<\beta_K$. Tras cada barrido de Glauber, actualizando en tablero de ajedrez, se propone intercambiar las configuraciones de dos temperaturas vecinas, aceptando con probabilidad dada por 

$$\min (1,\ e^{(\beta_{k+1}-\beta_k)\left(H(\sigma_{k+1})-H(\sigma_k)\right)} )$$

donde $H(\sigma)=\frac12\sum_{i\sim j}\big[\psi(j)-\psi(i)\big]^2$ y $\psi = \sigma - a$.

