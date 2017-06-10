# Derivatives, spectral anti-derivatives, and de-aliasing by discrete cosine 
# transforms in 2D. 
# Bryan Kaiser 
# 3/10/17

using PyPlot
using PyCall
using Base.FFTW


# =============================================================================
# readme:

# This script shows how to take 2D derivatives, do 2D spectral inversions, 
# and how to de-alias nonlinear signals using FFTW DSTs. At the end there is an
# additional Poisson equation solution for Gaussian on a linear slope 
# (e.g. a beta plane in GFD)

# cosine transforms are used for Neumann (derivative specified) boundary 
# conditions

# Make sure that you create the directory "/figures" in the same directory as 
# as this script, for output plots.


# =============================================================================
# choice of test signals

# test signal for derivative and inversion examples:
ifield = 2; # enter 1) for a 2D Gaussian signal or 2) for a 2D cosine wave

# test signal for de-aliasing example:
nonlinear_signal = 1; # enter 1) for 2D sine waves with noise or 2) for 2D 
# Gaussians with noise.


# =============================================================================

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
	# the same as the MATLAB function	
	m, n = length(vy), length(vx);
	vx = reshape(vx, 1, n); vy = reshape(vy, m, 1);
	(repmat(vx, m, 1), repmat(vy, 1, n));
end

function power_spectrum_2D(S::Array{Float64,2},Kmag::Array{Float64,2})
	# takes 2D spectra and generates a 1D power spectra for plotting. 
	# convert the gridded wavenumber magnitudes to a vector, remove 
	# repeated values, and sort:
	Kmag_vec_union = sort(union(vec(Kmag))); 
	S_vec = zeros(size(Kmag_vec_union)); # power spectrum
	for j = 1:length(Kmag_vec_union) # each wavenumber magnitude
		s = 0.0; count = 0.0;
		for n = 1:(size(Kmag,1)*size(Kmag,2)) # loop over Kmag, S
			if Kmag[n] == Kmag_vec_union[j];
				s = s + S[n];
				count = count+1.0;
			end
		end 
		S_vec[j] = s/count; # averaged magnitude
	end
	return S_vec, Kmag_vec_union;
end

function dealias(U::Array{Float64,2},V::Array{Float64,2},mask::Array{Float64,2})
	# 2/3 rule padding for de-aliasing a quadratic signal via dct
	return r2r(U.*mask,FFTW.REDFT01,1:2).*r2r(V.*mask,FFTW.REDFT01,1:2);
end

function poisson(q::Array{Float64,2},K::Array{Float64,2},L::Array{Float64,2})
	# 2D dct spectral inversion using FFTW to solve the Poisson equation:
	# given q and the Poisson equation, Laplacian(psi)=q, solve for psi.
	return r2r(-r2r(q,FFTW.REDFT10,1:2)./(4.0*Float64(size(q,1)*size(q,2))).*((K.^2+L.^2).^(-1.0)),FFTW.REDFT01,1:2); 
end


# =============================================================================
# domain
 
Lx = 3000.0; Ly = Lx; # m, domain size
Lxcenter = 0.0; Lycenter = 0.0; # x,y values @ the center of the grid
Nx = 2^7 # resolution, series length (must be at least even)
Ny = 2^7 # resolution, series length (must be at least even)
dx = Lx/Float64(Nx); dy = Ly/Float64(Ny);  # m, uniform grid spacings
x = collect(0.5*dx:dx:dx*Float64(Nx))-(Lx/2.0-Lxcenter); # m, centered uniform grid 
y = collect(0.5*dy:dy:dy*Float64(Ny))-(Ly/2.0-Lxcenter); # m, centered uniform grid
X,Y = meshgrid(x,y);


# =============================================================================
# choice of test signal for derivatives and spectral inversions

if ifield == 1 # A 2D Gaussian test case ("A" for analytical solutions)
sigma = Lx/20.0; psi = exp(-((X-Lxcenter).^2.0+(Y-Lycenter).^2.0)./(2.0*sigma^2.0));
div_psiA = (X-Lxcenter+Y-Lycenter).*psi.*(-sigma^(-2.0)); 
dpsidxA = -(X-Lxcenter).*psi.*(sigma^(-2.0)); 
dpsidyA = -(Y-Lycenter).*psi.*(sigma^(-2.0)); 
qA = psi.*(((X-Lxcenter).^2.0+(Y-Lycenter).^2.0).*sigma^(-4.0)-2.0*sigma^(-2.0)); 
elseif ifield == 2 # a 2D cosine wave
kx = (2.0*pi/Lx); ky = (2.0*pi/Ly); u = cos(X.*kx).*cos(Y.*ky);
divergenceA = -cos(X.*kx).*sin(Y.*ky).*ky-sin(X.*kx).*cos(Y.*ky).*kx;
dudxA =  -kx.*sin(X.*kx).*cos(Y.*ky); dudyA = -ky.*cos(X.*kx).*sin(Y.*ky);
laplacianA = -u.*(kx^(2.0)+ky^(2.0));
elseif ifield == 3 # a 2D cosine wave

end 


# =============================================================================
# plots of the signal:

fig = figure(); CP = contourf(X./Lx,Y./Ly,u,200,cmap="RdBu"); 
xlabel("x"); ylabel("y"); title("u, signal"); colorbar(CP); 
savefig("./figures/signal.png",format="png"); close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,dudxA,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("signal x derivative"); 
colorbar(CP); savefig("./figures/x_derivative_signal.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,dudyA,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("signal y derivative"); 
colorbar(CP); savefig("./figures/y_derivative_signal.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,divergenceA,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("signal divergence"); 
colorbar(CP); savefig("./figures/divergence_signal.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,laplacianA,200,cmap="PuOr");
xlabel("x"); ylabel("y"); title("Laplacian of signal"); 
colorbar(CP); savefig("./figures/Laplacian_signal.png",format="png"); 
close(fig);


# =============================================================================
# 2D discrete sine transform of psi(x,y) 

U = r2r(u,FFTW.REDFT10,1:2)./(4.0*Float64(Nx*Ny)); # DCT-II (2D)
Ux = r2r(u,FFTW.REDFT10,2)./(2.0*Float64(Nx)); # DCT-II (1D)
Uy = r2r(u,FFTW.REDFT10,1)./(2.0*Float64(Ny)); # DCT-II (1D)

uinv = r2r(U,FFTW.REDFT01,1:2); # DCT-III (inverse DST 2D)
idct_error = abs(u-uinv);

# inverse dst error
fig = figure(); CP = contourf(X./Lx,Y./Ly,idct_error,200,cmap="gray"); 
xlabel("x"); ylabel("y"); title("IDCT error"); colorbar(CP); 
savefig("./figures/idct_error.png",format="png"); close(fig);


# =============================================================================
# wavenumbers for derivatives/inversion

# sine transform wavenumbers
ks = collect(1:Nx).*(pi/Lx); # for DST-II
ls = collect(1:Ny).*(pi/Ly); # for DST-II
Ks,Ls = meshgrid(ks,ls); 
Kmags = (Ks.^2.0+Ls.^2.0).^(1.0/2.0); # gridded wavenumber magnitudes

# cosine transform wavenumbers
kc = collect(0:Nx-1).*(pi/Lx); # for DCT-II
lc = collect(0:Ny-1).*(pi/Ly); # for DCT-II
Kc,Lc = meshgrid(kc,lc); 
Kmagc = (Kc.^2.0+Lc.^2.0).^(1.0/2.0); # gridded wavenumber magnitudes
Kci = copy(Kc); Kci[1] = Inf; Lci = copy(Lc); Lci[1] = Inf;

# 1/length scale for plotting
Hkc = Kc./(2.0*pi); # 1/length
Hlc = Lc./(2.0*pi); # 1/length 
Hmagc = Kmagc./(2.0*pi);

# wavenumber plots:

fig = figure(); CP = contourf(1:Nx,1:Ny,Kmags,200,cmap="spectral"); 
xlabel("1:N"); ylabel("1:N"); title("|K| sine"); colorbar(CP); 
savefig("./figures/sine_wavenumber_magnitude.png",format="png"); close(fig);

fig = figure(); CP = contourf(0:Nx-1,0:Ny-1,Kmagc,200,cmap="spectral"); 
xlabel("0:N-1"); ylabel("0:N-1"); title("|K| cosine"); colorbar(CP); 
savefig("./figures/cosine_wavenumber_magnitude.png",format="png"); close(fig);


# =============================================================================
# divergence by discrete sine transform of psi(x,y)

# first derivatives by dst
Ushiftx = zeros(size(Kmagc)); Ushiftx[:,1:Int64(Nx-1)] = Ux[:,2:Int64(Nx)]; 
Ushifty = zeros(size(Kmagc)); Ushifty[1:Int64(Ny-1),:] = Uy[2:Int64(Ny),:]; 
dudx = r2r(-Ushiftx.*Ks,FFTW.RODFT01,2); # DST-III (inverse dst)
dudy = r2r(-Ushifty.*Ls,FFTW.RODFT01,1); # DST-III (inverse dst)
divergence = dudx+dudy;

# divergence error
divergence_error = abs(divergenceA-divergence); 
max_div_error = maximum(divergence_error);
println("The maximum divergence computation error is $(max_div_error) for a $Nx by $Ny grid.\n")

# first derivative plots:

fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(dudxA-dudx),200,cmap="gray");
xlabel("x"); ylabel("y"); title("x derivative, error"); 
colorbar(CP); savefig("./figures/x_derivative_error.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,abs(dudyA-dudy),200,cmap="gray");
xlabel("x"); ylabel("y"); title("y derivative, error"); 
colorbar(CP); savefig("./figures/y_derivative_error.png",format="png"); 
close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,divergence_error,200,cmap="gray");
xlabel("x"); ylabel("y"); title("divergence, error"); 
colorbar(CP); savefig("./figures/divergence_error.png",format="png"); 
close(fig);


# =============================================================================
# Laplacian by discrete sine transform of psi(x,y)

# the Laplacian by inverse dst
laplacian = r2r(-Ux.*Kc.^2.0,FFTW.REDFT01,2)+r2r(-Uy.*Lc.^2.0,FFTW.REDFT01,1);

# maximum error
laplacian_error = abs(laplacianA-laplacian); max_lap_error = maximum(laplacian_error);
println("The maximum Laplacian computation error is $(max_lap_error) for a $Nx by $Ny grid.\n")

# plot of the real component, computational error 
fig = figure(); CP = contourf(X./Lx,Y./Ly,laplacian_error,200,cmap="gray");
xlabel("x"); ylabel("y"); title("Laplacian, error"); 
colorbar(CP); savefig("./figures/Laplacian_error.png",format="png"); 
close(fig);


# =============================================================================
# De-aliasing a nonlinear (quadratic) signal

if nonlinear_signal == 1 # sine waves with random noise
kx = (2.0*pi/Lx); ky = (2.0*pi/Ly); 
ua = rand(size(X)).*0.5+cos(X.*kx).*cos(Y.*ky);
ub = rand(size(X)).*0.5+cos(X.*kx).*cos(Y.*ky);
elseif nonlinear_signal == 2 # Gaussian with random noise
sigma = Lx/10.0; 
ua = exp(-((X-Lxcenter).^2+(Y-Lycenter).^2)./(2.0*sigma^2))+rand(size(X)).*0.5;
ub = exp(-((X-Lxcenter).^2+(Y-Lycenter).^2)./(2.0*sigma^2))+rand(size(X)).*0.5;
end

# cosine transform
Ua = r2r(ua,FFTW.REDFT10,1:2)./(4.0*Float32(Nx*Ny)); # DST-II 
Ub = r2r(ub,FFTW.REDFT10,1:2)./(4.0*Float32(Nx*Ny)); # DST-II

# mask for 2/3 rule padding for de-aliasing a quadratic signal via fft
mask = ones(size(Kmagc));
for j = 1:(size(Kmagc,1)*size(Kmagc,2)) # 2/3 rule
	if abs(Kmagc[j]) >= max(size(Kmagc,1),size(Kmagc,2))/3.0*Kmagc[2]; 
		mask[j] = 0.0; # dct space variables
	end
end

# aliased and de-aliased quadratic signal
u2_alias = ua.*ub; # aliased square
tic(); u2_dealias = dealias(Ua,Ub,mask); time = toq(); # de-aliased square
println("The computation time is for the de-aliased signal is $time seconds for a $Nx by $Ny grid.\n")

# sine transform of quadratic signal
S_alias = r2r(u2_alias,FFTW.RODFT10,1:2)./(4.0*Float32((Nx^2.0+Ny^2.0)^(1.0/2.0))); 
S1 = (abs(S_alias)); #.*2.0/float(N); 
S_dealias = r2r(u2_dealias,FFTW.RODFT10,1:2)./(4.0*Float32((Nx^2.0+Ny^2.0)^(1.0/2.0))); 
S2 = (abs(S_dealias)); #.*2.0/float(N); 

# 1D power spectrum of 2D quadratic signals
S1_vec, Hmag_vec1 = power_spectrum_2D(S1,Hmagc);
S2_vec, Hmag_vec2 = power_spectrum_2D(S2,Hmagc);

# plots:

fig = figure(); semilogx(Hmag_vec1,S1_vec,"r",label="aliased");
semilogx(Hmag_vec2,S2_vec,"b",label="de-aliased"); legend();
xlabel("k"); ylabel("|PSI|"); title("2D power spectrum"); 
axis([minimum(Hmag_vec1),maximum(Hmag_vec1),0.0,maximum(S1_vec)*1.2]);
savefig("./figures/quadratic_signal_power_spectrum.png",format="png"); close(fig);

fig = figure(); CP = contourf(X./Lx,Y./Ly,u2_alias,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("u*u aliased"); 
colorbar(CP); savefig("./figures/quadratic_signal_aliased.png",format="png"); 

fig = figure(); CP = contourf(X./Lx,Y./Ly,u2_dealias,200,cmap="PuOr"); 
xlabel("x"); ylabel("y"); title("u*u de-aliased"); 
colorbar(CP); savefig("./figures/quadratic_signal_dealiased.png",format="png");


# =============================================================================
# Poisson equation solution by dst

# Poisson equation solution: Laplacian(psi) = qA
tic(); uP = poisson(laplacianA,Kci,Lci); time = toq(); 
println("The first Poisson equation computation time is $(time) seconds for a $Nx by $Ny grid.\n")
Poisson_error = abs(uP-u); # Poisson equation solution error
max_err = maximum(Poisson_error);

fig = figure(); CP = contourf(X./Lx,Y./Ly,uP,200,cmap="RdBu");
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q, psi solution"); 
colorbar(CP); savefig("./figures/Poisson_solution.png",format="png"); 

fig = figure(); CP = contourf(X./Lx,Y./Ly,Poisson_error,200,cmap="gray")
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q, psi solution error"); 
colorbar(CP); savefig("./figures/Poisson_solution_error.png",format="png"); 

println("The maximum Poisson equation computation error is $(max_err) for a $Nx by $Ny grid.\n")

# Another example: a Gaussian on a linear y slope (beta plane):
sigma = Lx/20.0; beta = 1E-9;
psi2 = exp(-((X-Lxcenter).^2.0+(Y-Lycenter).^2.0)./(2.0*sigma^2.0));
qA2 = psi2.*(((X-Lxcenter).^2.0+(Y-Lycenter).^2.0).*sigma^(-4.0)-2.0*sigma^(-2.0))-Y.*beta; 

# Poisson equation solution: Laplacian(psi) = qA
tic(); psiP2 = poisson(qA2,Kci,Lci); time = toq(); 
println("The second Poisson equation computation time is $time seconds for a $Nx by $Ny grid.\n")

fig = figure(); CP = contourf(X./Lx,Y./Ly,psiP2,200,cmap="Spectral");
xlabel("x"); ylabel("y"); title("Laplacian(psi) = q-By, psi solution"); 
colorbar(CP); savefig("./figures/Poisson_solution_linear_slope.png",format="png");

