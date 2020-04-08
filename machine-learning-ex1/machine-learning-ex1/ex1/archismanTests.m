data = load('ex1data2.txt');
X = data(:,[1 2]);
Y = data(:,3);


% feature normalization starts here , The code henceforth is meant to be 
% independant of the number of variables

mu = zeros(1, size(X,2));
sigma = zeros(1,size(X,2));

mu = mean(X);
sigma = std(X);

X_norm = (X-mu)./sigma;