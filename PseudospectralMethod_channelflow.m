%-------------------------------------------------------------------------%
% Simulating pressure driven flow using Fourier-Chebyshev collocation
%-------------------------------------------------------------------------%
% Domain:                      dP/dx|m = -2/Re
% y=1    |---------------------------------------------------|
%        |                --->                               |
%        |                  --->   (Flow direction)          |
%        |                  --->                             |
%        |                --->                               |
% y=-1   |---------------------------------------------------|
%        x=0                                               x=2*pi
% No slip boundary condition at y=-1,+1. Periodic boundary condition in x
%
%---------------------------Solver Details--------------------------------%
% Fourier collocation in x and chebyshev collocation in y.
% 2nd order Adams Bashforth scheme for explicit treatment of nonlinear
% terms. 2nd order backward differencing scheme for implicit treatment of 
% linear terms


%% Clean
clc;clear all;close all;warning('off');

%% Grid,Coordinates
tic
Nx =8;Ny=26; Lx= 2*pi;Ly=2;
y = -cos(pi*(0:Ny)/Ny)';
ys  = -cos(pi*(0.5:Ny-0.5)/Ny)';
x  = (0:Nx-1)/Nx*2*pi;

%% Modes

kx  = fftshift(-Nx/2:Nx/2-1);    % wave number vector

Re=100;vis = 1/Re; % Re-Reynolds number, visc = 1/Re

[X,Y]   = meshgrid(x,y);
[Xs,Ys] = meshgrid(x,ys);
% Chebyshev Differentiation Matrices for Gauss-Lobatto & Gauss
T = cos(acos(y).*(0:Ny));
D = cheb(Ny,y,1);
D2etagl = D^2;
% Altering the cheb function for finding differentiation matrix
% of the staggered chebyshev points did not work, hence calculated 
% by finding derivative of chebyshev polynomial
%Ds = cheb(Ny,ys,2);
Ts  = cos(acos(ys)*(0:Ny-1));
DTs = diag(1./sqrt(1-ys.^2))*sin(acos(ys)*(0:Ny-1))*diag(0:Ny-1);
Ds = DTs/Ts;
% Time
tfinal = 4;  dt = 1e-3;  Nstep  = ceil(tfinal/dt);  t      = (0:Nstep)*dt;

% Other operators
ZNx = diag(ones(1,Ny+1));
ZNz = ZNx;
ZNy = diag([0 ones(1,Ny-1) 0]);
Igl = eye(Ny+1);
Ig  = eye(Ny);
Igldt=(3/(2*dt))*Igl; %For time stepping
Iglvis=vis*Igl; %For x-viscous term
% Divergence and gradient operators
G_trunc = [Ts zeros(Ny,1)]; %Padded with 0 as staggered grid has Ny-1 elements
div_x  = G_trunc*(T\(Igl*ZNx));
div_y  = G_trunc*(T\(D*ZNy)); %divided by T and multiplied by Ts for interpolating
grad_x = T(:,1:Ny)*(Ts\Ig); %divided by Ts and multiplied by T for interpolating
grad_y = T(:,1:Ny)*(Ts\Ds); 
div_x_act_on_grad_x = -Ig;
div_y_act_on_grad_y = div_y*grad_y;

% Test Functions U,V,P
% ufun = @(x,y,z,t) Ly-(y-Ly/2).^2;
ufun = @(x,y,t) zeros(size(x));
vfun = @(x,y,t) x*0;
Pfun = @(x,y,t) (-2/Re)*ones(size((x)));
% Initial conditions,
u=ufun(X,Y,0); uold=ufun(X,Y,-dt);
v=vfun(X,Y,0); vold=vfun(X,Y,-dt);
Pm=Pfun(X,Y,0);
[unewh,vnewh,dudy,dvdy,dudyold,dvdyold]=deal(zeros(size(u)));[pnewh,continuity]=deal(zeros(size(Ys)));
% Initial conditions, fft'ed
Ph = fft(Pm,[],2);
tol=1e-6;     % tolerance-> steady & continuity

tic
for j=1:Nstep

    uh = fft(u,[],2); vh = fft(v,[],2);
    uoldh = fft(uold,[],2);    voldh = fft(vold,[],2);
    %% Exact Convective Term Functions (Comment the ones to compare)
    %% Convective Terms Approximations
    %%%%%%%%%Convective Terms%%%%%%%%%
    % u*dudx & v*dudx (at time t & t-1) - non dealiased
    dudxh= bsxfun(@times, kx*1i, uh);    dudxoldh= bsxfun(@times, kx*1i, uoldh);
    dvdxh= bsxfun(@times, kx*1i, vh);    dvdxoldh= bsxfun(@times, kx*1i, voldh);
    ududxh=pseudo(uh,dudxh);udvdxh=pseudo(uh,dvdxh);
    ududxoldh=pseudo(uoldh,dudxoldh);udvdxoldh=pseudo(uoldh,dvdxoldh);

    %% v*dudy & v*dvdy (at time t & t-1) - dealiased
    dudy = D*u; dvdy = D*v;
    dudyold = D*uold; dvdyold = D*vold;

    vdudyh = pseudo(vh,fft(dudy,[],2));
    vdvdyh = pseudo(vh,fft(dvdy,[],2));
    vdudyoldh = pseudo(voldh,fft(dudyold,[],2));
    vdvdyoldh = pseudo(voldh,fft(dvdyold,[],2));

    % Loop over Fourier modes


    for k=1:length(kx)
        % Solving for intermediate velocities; ustar & vstar
        A=Igldt-D2etagl*vis+(kx(k)^2)*Iglvis;

        uconv=2*ududxh(:,k)+2*vdudyh(:,k)-ududxoldh(:,k)-vdudyoldh(:,k);
        rhsuint=(2/dt)*uh(:,k)-(1/(2*dt))*uoldh(:,k)+uconv-Ph(:,k);
        ustar       = A\rhsuint;
        %ustar([1 Ny+1]) = 0; %no slip BC
        vconv=2*vdvdyh(:,k)+2*udvdxh(:,k)-udvdxoldh(:,k)-vdvdyoldh(:,k);
        rhsvint=(2/dt)*vh(:,k)-(1/(2*dt))*voldh(:,k)+vconv;
        vstar       = A\rhsvint;
        vstar([1 Ny+1])=0; %no slip BC

        % solve pressure poisson eqn L*q=f;
        RHS = div_x*(kx(k)*1i)*ustar + div_y*vstar; %Fhat Eqn 7.3.72
        %LHS = dt*(grad_x*kx(k)*1i + grad_y);
        LHS = dt*( div_x_act_on_grad_x*(kx(k)^2) + div_y_act_on_grad_y ); %L Eqn 7.3.71
        pnewh(:,k) = LHS\RHS;
        % Update velocities
        unewh(:,k) = ustar-dt*grad_x*(kx(k)*1i)*pnewh(:,k);
        unewh([1 Ny+1],k) = ustar([1 Ny+1]) + dt*div_x_act_on_grad_x([1 Ny],[1 Ny])*(kx(k)^2)*pnewh([1 Ny],k);
        vnewh(:,k) = vstar-dt*grad_y*pnewh(:,k);
        continuity(:,k)=div_x*(kx(k)*1i)*unewh(:,k)+div_y*vnewh(:,k); %to check for continuity

    end

    unew = real(ifft(unewh,[],2));
    vnew = real(ifft(vnewh,[],2));
    % Enforce BC
    unew([1 Ny+1],:) = ufun(X([1 Ny+1],:),Y([1 Ny+1],:),t(j+1));    % B.C.
    vnew([1 Ny+1],:) = vfun(X([1 Ny+1],:),Y([1 Ny+1],:),t(j+1));    % B.C.
    steady=(1/(2*dt))*(3*unew-4*u+uold);s=max(abs(steady(:)));
    if s<tol
        disp('solution stedy-tol 1e-6');
        save steadymapped;
        break;
    end

    uold = real(ifft(uh,[],2));
    vold = real(ifft(vh,[],2));
    u=unew;v=vnew;
    rescont=sum(continuity(:));
    if rescont>1e-10,disp('Cont not satisfied');break;end
    if mod(j,100)==0
        U(:,:,j/100) = u;
    end%disp(num2str(j/Nstep)); filename=['Nx',num2str(Nx),'Ny',num2str(Ny),'Lx',num2str(Lx),'Ly',num2str(Ly),'Re',num2str(Re),'dt',num2str(dt),'T',num2str(t(j)),'BDFAB2.mat'];	save(filename,'Nx','Ny','Re','j','Nstep','u','v','X','Y', 'dt','rescont','Lx','Ly','s');end
end
toc
%% Plotting
contourf(X/pi,Y,u,'edgecolor','none')
xlabel('x (1/$\pi$ units)')
ylabel('y')
title(sprintf('t = %f',dt*j))
cb=colorbar();
ylabel(cb,'u (x-vel component)','Rotation',90)
% figure(2)
% fig=ceil(Nstep/5):ceil(Nstep/5):Nstep;
% for i=1:length(fig)
%     subplot(1,length(fig),i)
%     contourf(X/pi,Y,U(:,:,fig(i)),'edgecolor','none');
%     xlabel('x (1/$\pi$ units)')
%     ylabel('y')
%     title(sprintf('t = %f',dt*fig(i)))
%     cb=colorbar();
%     ylabel(cb,'u (x-vel component)','Rotation',90)
% end
%% Functions
%Pseudospectral calculation of nonlinear terms
function ph=pseudo(uhat,vhat)
% in: uh,vh from fft with n samples
[ny,nx]=size(uhat);m=nx*3/2;
uhp=[uhat(:,1:nx/2) zeros(ny,(m-nx)) uhat(:,nx/2+1:nx)]; % pad uhat with zeros
vhp=[vhat(:,1:nx/2) zeros(ny,(m-nx)) vhat(:,nx/2+1:nx)]; % pad vhat with zeros
up=ifft(uhp,[],2); vp=ifft(vhp,[],2); w=up.*vp; wh=fft(w,[],2);
ph=1.5*[wh(:,1:nx/2) wh(:,m-nx/2+1:m)]; % extract F-coefficients
end
%Physical space to Fourier-Chebyshev space
function uhatc = fct(u)

Nx = size(u,2);
uhat = fft(u,[],2)/Nx;

uhatc = ChebTrans(uhat);

if (mod(Nx,2) == 0)
    uhatc(:,Nx/2+1) = 0;
end

end
%Fourier-Physical to Fourier-Chebyshev space
function uhatc = ChebTrans(uhat)

Nx = size(uhat,2);
Ny = size(uhat,1);

uhatc = uhat;

N = Ny-1;
v = zeros(2*Ny - 2,Nx);
v(1:Ny,:) = uhatc(end:-1:1,:);
v(Ny+1:end,:)= uhatc(2:N,:);

a = fft(v,[],1)/N;
uhatc(1,:) = a(1,:)/2;
uhatc(2:N,:) = a(2:N,:);
uhatc(N+1,:) = a(N+1,:)/2;

if (mod(Nx,2) == 0)
    uhatc(:,Nx/2+1) = 0;
end

end

%Fourier-Chebyshev space to Physical space
function u = ifct(uhatc)

Nx = size(uhatc,2);
Ny = size(uhatc,1);
N = Ny - 1;

uhat = iChebTrans(uhatc);
u = ifft(uhat,[],2)*Nx;
end

%Fourier-Chebyshev tot Fourier-Physical space
function uhat = iChebTrans(uhatc)

Nx = size(uhatc,2);
Ny = size(uhatc,1);
N = Ny - 1;

if (mod(Nx,2) == 0)
    uhatc(:,Nx/2+1) = 0;
end

uhatc(1,:) = uhatc(1,:)*2;
uhatc(2:N,:) = uhatc(2:N,:);
uhatc(end,:) = uhatc(end,:)*2;

M = zeros(2*Ny-2,Nx);
M(1:Ny,:) = uhatc;
M(Ny+1:end,:) = uhatc(end-1:-1:2,:);
v = ifft(M,[],1);
uhat = v(N+1:-1:1,:)*N;
end
function [D] = cheb(N,x,o)
%if N==0, D=0; x=1; return, end
%x = cos(pi*(0:N)/N)';
if o==1
    c = [2; ones(N-1,1); 2].*(-1).^(0:N)';
    X = repmat(x,1,N+1);
    dX = X-X';
    D  = (c*(1./c)')./(dX+(eye(N+1)));
    D  = D - diag(sum(D'));
elseif o==2
    c = [2; ones(N-2,1); 2].*(-1).^(0.5:N)';
    X = repmat(x,1,N);
    dX = X-X';
    D  = (c*(1./c)')./(dX+(eye(N)));
    D  = D - diag(sum(D'));
end


end