points = [3,4,5,6,7];
%sim = 'Esplit_%s%i_scaling1.txt';
sim = 'l0_%s%i_7.5GeV_2.txt';
logYplot = 1;
logXplot = 1;

figure(5)
co = get(gca,'ColorOrder');
for i=1:length(points)
    model = readtable(sprintf(sim,'model',points(i)));
    [model_mu,model_sigma] = normal2lognormal(model.Var2,model.Var3);
    example = readtable(sprintf(sim,'example',points(i)));
    [example_mu,example_sigma] = normal2lognormal(example.Var2,example.Var3);
    acq = readtable(sprintf(sim,'acq',points(i)));

    subplot(2,length(points),i)
    if logYplot
        shadedErrorBar(model.Var1,model.Var2/log(10),model.Var3/log(10))
    else
        shadedErrorBar(model.Var1,model_mu,model_sigma)
    end
    hold on
    if logYplot
        errorbar(example.Var1,example.Var2/log(10),example.Var3/log(10),'o','Color',co(1,:),'MarkerFaceColor',co(1,:),'MarkerSize',4)
        if i>1 && points(i)>points(i-1)+1
            diff = points(i)-points(i-1)-1;
            errorbar(example.Var1(end-diff:end),example.Var2(end-diff:end)/log(10),example.Var3(end-diff:end)/log(10),'o','Color',co(4,:),'MarkerFaceColor',co(4,:),'MarkerSize',4)
        else
           errorbar(example.Var1(end),example.Var2(end)/log(10),example.Var3(end)/log(10),'o','Color',co(4,:),'MarkerFaceColor',co(4,:),'MarkerSize',4)
        end
    else
        errorbar(example.Var1,example_mu,example_sigma,'o','Color',co(1,:),'MarkerFaceColor',co(1,:),'MarkerSize',4)
        if i>1 && points(i)>points(i-1)+1
            diff = points(i)-points(i-1)-1;
            errorbar(example.Var1(end-diff:end),example_mu(end-diff:end),example_sigma(end-diff:end),'o','Color',co(4,:),'MarkerFaceColor',co(4,:),'MarkerSize',4)
        else
            errorbar(example.Var1(end),example_mu(end),example_sigma(end),'o','Color',co(4,:),'MarkerFaceColor',co(4,:),'MarkerSize',4)
        end
    end
    if logXplot
        set(gca,'XScale','log')
    end
    hold off
    title(sprintf('N=%i',points(i)))
    set(gca,'XTickLabel',[])
    if i==1
        if logYplot
            ylabel('log_{10} N_+/N+-')
        else
            ylabel('N_+/N+-')
        end
    end

    subplot(4,length(points),2*length(points)+i)
    %semilogy(acq.x_,acq.EsplitEI,'Color',co(2,:))
    %xlabel('s=E_{coll}/E_{tot}')
    %semilogy(acq.x_,acq.l0EI,'Color',co(2,:))
    plot(acq.x_,acq.l0EI,'Color',co(2,:)) % using a log(acq) function
    xlabel('l_0 (m)')
    if logXplot
        set(gca,'XScale','log')
    end
    if i==1
        ylabel('E.I.')
    end
end
