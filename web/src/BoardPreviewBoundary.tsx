import React from 'react';
export class BoardPreviewBoundary extends React.Component<{children:React.ReactNode;resetKey:string},{error:string|null}>{
 state:{error:string|null}={error:null};
 static getDerivedStateFromError(error:Error){return {error:error.message||'Board preview failed'};}
 componentDidUpdate(previous:Readonly<{children:React.ReactNode;resetKey:string}>){if(previous.resetKey!==this.props.resetKey&&this.state.error)this.setState({error:null});}
 render(){return this.state.error?<div className="preview-error" role="alert"><strong>Board preview unavailable</strong><p>The camera can continue tracking. Retry the preview or download the latest recognition from More.</p><p>{this.state.error}</p><button onClick={()=>this.setState({error:null})}>Retry preview</button></div>:this.props.children;}
}
