% Plots a distribution of numbers for seeing how random they are

x = [0; 1; 2];
y = zeros(3,1);

for i = 1:10000
	random_test = floor(3*rand(1));
	y(random_test + 1) = y(random_test + 1) + 1;
end

x
y

plot(x,y)

