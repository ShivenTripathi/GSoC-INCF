function H = Entropy_millermaddow(ns,K)

nTot = sum(sum(ns));
p = ns/nTot;
H1 = -nansum(nansum(p.*log2(p)));
m = K-length(ns);   % len changed to length
H = H1+((m-1)/2/nTot);