const cases=document.querySelectorAll('.case');const btns=document.querySelectorAll('#filterbar button');const search=document.getElementById('qsearch');
let curFilter='all',curSearch='';
function apply(){cases.forEach(c=>{let show=true;
  if(curFilter!=='all'&&c.dataset.root!==curFilter)show=false;
  if(show&&curSearch)show=c.dataset.text.includes(curSearch.toLowerCase());
  c.style.display=show?'':'none';});}
btns.forEach(b=>b.addEventListener('click',()=>{btns.forEach(x=>x.classList.remove('active'));b.classList.add('active');curFilter=b.dataset.filter;apply();}));
search.addEventListener('input',e=>{curSearch=e.target.value;apply();});
