import React from "react";
import { Link } from "react-router-dom";

//at the top of the page
function Header() {
  return (
    //design
    <header
      style={{
        padding: "1rem 1.5rem",
        borderBottom: "1px solid #ddd",
        marginBottom: "1rem",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      //links
      <Link to="/" style={{ textDecoration: "none", color: "#222" }}>
        <h1 style={{ margin: 0, fontSize: "1.4rem" }}>Macroroni</h1>
      </Link>
      <nav style={{ display: "flex", gap: "1rem" }}>
        <Link to="/" style={{ textDecoration: "none", color: "#555" }}>
          Home
        </Link>
        <Link to="/docs" style={{ textDecoration: "none", color: "#555" }}>
          Docs
        </Link>
        <Link to="/playground" style={{ textDecoration: "none", color: "#555" }}>
          Playground
        </Link>
      </nav>
    </header>
  );
}


export default Header;
