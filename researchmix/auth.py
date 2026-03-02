# researchmix/auth.py
import streamlit as st
from researchmix.config import get_env
from researchmix.state import u_bucket


def render_login_gate() -> bool:
    """
    Returns True if logged in, else renders login UI and returns False.
    """
    if st.session_state.auth.get("logged_in"):
        return True

    env = get_env()
    demo_users = env["DEMO_USERS"]

    st.title("🔐 ResearchMix Login")
    st.caption("Hackathon demo login (dummy auth).")

    with st.container(border=True):
        username = st.text_input("Username", placeholder="demo / judge / fairgame")
        password = st.text_input("Password", type="password", placeholder="demo / demo / researchmix")
        c1, c2 = st.columns([1, 1])

        with c1:
            if st.button("Log in", type="primary", use_container_width=True):
                if username in demo_users and demo_users[username] == password:
                    st.session_state.auth["logged_in"] = True
                    st.session_state.auth["username"] = username
                    u_bucket()  # ensure bucket exists
                    st.success("Logged in ✅")
                    st.rerun()
                else:
                    st.error("Invalid username/password.")

        with c2:
            if st.button("Use demo account", use_container_width=True):
                st.session_state.auth["logged_in"] = True
                st.session_state.auth["username"] = "demo"
                u_bucket()
                st.success("Logged in as demo ✅")
                st.rerun()

    st.info("Tip for judges: use **judge / demo** or click **Use demo account**.")
    return False